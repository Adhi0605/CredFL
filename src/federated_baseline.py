"""Train and evaluate the Week 3 plain federated-learning baseline."""

from __future__ import annotations

import argparse
import copy
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.centralized_baseline import (
    FraudMLP,
    MLPConfig,
    _validate_frame,
    evaluate_mlp,
    select_f1_threshold,
    set_reproducible_seed,
    to_tensors,
)
from src.data import FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocessing import FraudPreprocessor


@dataclass(frozen=True)
class FederatedConfig:
    """Reproducible settings for synchronous, full-participation FedAvg."""

    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout: float = 0.2
    rounds: int = 10
    local_epochs: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 2048
    weight_decay: float = 1e-4
    validation_fraction: float = 0.1
    seed: int = 42


def load_client_frames(data_dir: str | Path) -> tuple[list[pd.DataFrame], list[Path]]:
    paths = sorted(Path(data_dir).glob("client_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No client_*.csv files found in {data_dir}.")
    frames = [pd.read_csv(path) for path in paths]
    for path, frame in zip(paths, frames, strict=True):
        _validate_frame(frame, path)
    return frames, paths


def split_client_frames(
    frames: Sequence[pd.DataFrame], validation_fraction: float, seed: int
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Keep each client's validation rows out of every local update."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    training_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    for client_id, frame in enumerate(frames):
        if frame[TARGET_COLUMN].nunique() != 2:
            raise ValueError(f"Client {client_id} must contain both classes for stratification.")
        training, validation = train_test_split(
            frame,
            test_size=validation_fraction,
            random_state=seed + client_id,
            stratify=frame[TARGET_COLUMN],
        )
        training_frames.append(training.reset_index(drop=True))
        validation_frames.append(validation.reset_index(drop=True))
    return training_frames, pd.concat(validation_frames, ignore_index=True)


def weighted_fedavg(
    client_states: Sequence[Mapping[str, torch.Tensor]], client_sizes: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Return the example-count-weighted average of client model states."""
    if not client_states or len(client_states) != len(client_sizes):
        raise ValueError("client_states and client_sizes must be non-empty and equally sized.")
    if any(size <= 0 for size in client_sizes):
        raise ValueError("Every participating client must have at least one example.")
    keys = tuple(client_states[0].keys())
    if any(tuple(state.keys()) != keys for state in client_states[1:]):
        raise ValueError("All client states must contain the same parameters in the same order.")

    total_size = float(sum(client_sizes))
    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        reference = client_states[0][key]
        accumulator = torch.zeros_like(reference, dtype=torch.float64, device="cpu")
        for state, size in zip(client_states, client_sizes, strict=True):
            value = state[key]
            if value.shape != reference.shape:
                raise ValueError(f"Mismatched shape for parameter {key}.")
            accumulator.add_(value.detach().to(device="cpu", dtype=torch.float64), alpha=size)
        averaged[key] = (accumulator / total_size).to(dtype=reference.dtype)
    return averaged


def train_local_model(
    global_model: FraudMLP,
    frame: pd.DataFrame,
    config: FederatedConfig,
    global_pos_weight: float,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], float]:
    """Train one client from the current global state without privacy mechanisms."""
    set_reproducible_seed(seed)
    model = copy.deepcopy(global_model).to(device)
    features, labels = to_tensors(frame)
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(global_pos_weight, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_loss = 0.0
    observations = 0
    model.train()
    for _ in range(config.local_epochs):
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_labels)
            observations += len(batch_labels)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return state, total_loss / observations


def train_federated(
    client_frames: Sequence[pd.DataFrame], config: FederatedConfig, device: torch.device
) -> tuple[FraudMLP, list[dict]]:
    """Run synchronous full-participation FedAvg over all supplied clients."""
    if not client_frames:
        raise ValueError("At least one client frame is required.")
    set_reproducible_seed(config.seed)
    global_model = FraudMLP(
        len(FEATURE_COLUMNS), config.hidden_sizes, config.dropout
    ).to(device)
    fraud_count = sum(int(frame[TARGET_COLUMN].sum()) for frame in client_frames)
    row_count = sum(len(frame) for frame in client_frames)
    if fraud_count == 0 or fraud_count == row_count:
        raise ValueError("Federated training data must contain both classes.")
    global_pos_weight = (row_count - fraud_count) / fraud_count
    client_sizes = [len(frame) for frame in client_frames]
    history: list[dict] = []

    for round_number in range(1, config.rounds + 1):
        client_states: list[dict[str, torch.Tensor]] = []
        client_losses: list[float] = []
        for client_id, frame in enumerate(client_frames):
            state, loss = train_local_model(
                global_model,
                frame,
                config,
                global_pos_weight,
                device,
                config.seed + round_number * 10_000 + client_id,
            )
            client_states.append(state)
            client_losses.append(loss)
        global_model.load_state_dict(weighted_fedavg(client_states, client_sizes))
        weighted_loss = float(np.average(client_losses, weights=client_sizes))
        history.append(
            {
                "round": round_number,
                "weighted_training_loss": weighted_loss,
                "client_losses": client_losses,
                "participating_clients": len(client_frames),
            }
        )
        print(
            f"Round {round_number:02d}/{config.rounds}: "
            f"weighted_loss={weighted_loss:.6f}"
        )
    return global_model, history


def write_comparison_report(
    federated_metrics: Mapping[str, float | int],
    centralized_metrics: Mapping[str, float | int],
    config: FederatedConfig,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = (
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("AUC-ROC", "roc_auc"),
        ("Average precision", "average_precision"),
    )
    lines = [
        "# Week 3 Plain Federated-Learning Results",
        "",
        f"Six clients participated in every one of {config.rounds} synchronous rounds,",
        f"training for {config.local_epochs} local epoch(s) per round. The server applied",
        "sample-count-weighted FedAvg. No clipping, noise, differential privacy, or",
        "secure aggregation was used. The decision threshold came only from held-out",
        "client-training rows; `global_test.csv` remained evaluation-only.",
        "",
        "| Metric | Centralized | Plain FedAvg | Difference (FL - centralized) |",
        "|---|---:|---:|---:|",
    ]
    for label, key in metric_rows:
        centralized = float(centralized_metrics[key])
        federated = float(federated_metrics[key])
        lines.append(
            f"| {label} | {centralized:.6f} | {federated:.6f} | "
            f"{federated - centralized:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"FedAvg decision threshold: {float(federated_metrics['threshold']):.6f}.",
            f"Predicted fraud cases: {int(federated_metrics['predicted_fraud_count']):,} ",
            f"of {int(federated_metrics['test_rows']):,} test rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds <= 0 or args.local_epochs <= 0 or args.batch_size <= 0:
        raise ValueError("rounds, local_epochs, and batch_size must be positive.")
    config = FederatedConfig(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(device_name)

    raw_clients, client_paths = load_client_frames(args.data_dir)
    raw_training_clients, raw_validation = split_client_frames(
        raw_clients, config.validation_fraction, config.seed
    )
    test_raw = pd.read_csv(args.data_dir / "global_test.csv")
    _validate_frame(test_raw, args.data_dir / "global_test.csv")
    preprocessor = FraudPreprocessor.load(args.artifacts_dir / "preprocessor.json")
    training_clients = [preprocessor.transform(frame) for frame in raw_training_clients]
    validation = preprocessor.transform(raw_validation)
    test = preprocessor.transform(test_raw)

    model, history = train_federated(training_clients, config, device)
    threshold_config = MLPConfig(
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
        batch_size=config.batch_size,
        threshold=0.5,
        seed=config.seed,
    )
    selected_threshold, validation_f1 = select_f1_threshold(
        model, validation, threshold_config, device
    )
    metrics = evaluate_mlp(model, test, selected_threshold, config.batch_size, device)

    centralized_run_path = args.artifacts_dir / "centralized_mlp_run.json"
    centralized_run = json.loads(centralized_run_path.read_text(encoding="utf-8"))
    run_metadata = {
        "model": "FraudMLP",
        "federated_algorithm": "sample-weighted FedAvg",
        "privacy_mechanism": None,
        "aggregation": {
            "method": "FedAvg",
            "weighting": "client fit-row count",
            "client_participation": "all clients in every round",
        },
        "input_features": FEATURE_COLUMNS,
        "config": asdict(config),
        "device": str(device),
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "client_files": [path.name for path in client_paths],
        "client_fit_rows": [len(frame) for frame in training_clients],
        "client_fit_fraud_counts": [
            int(frame[TARGET_COLUMN].sum()) for frame in training_clients
        ],
        "total_fit_rows": sum(len(frame) for frame in training_clients),
        "validation_rows": len(validation),
        "threshold_selection": {
            "method": "maximum F1 on per-client stratified validation splits",
            "selected_threshold": selected_threshold,
            "validation_f1": validation_f1,
        },
        "history": history,
        "metrics": metrics,
        "centralized_metrics": centralized_run["metrics"],
    }

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.artifacts_dir / "federated_mlp.pt")
    (args.artifacts_dir / "federated_mlp_run.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    write_comparison_report(
        metrics,
        centralized_run["metrics"],
        config,
        args.reports_dir / "week3_federated_metrics.md",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved Week 3 artifacts to {args.artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
