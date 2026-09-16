"""Train and evaluate the Week 2 centralized MLP fraud baseline."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import EXPECTED_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocessing import FraudPreprocessor


@dataclass(frozen=True)
class MLPConfig:
    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 2048
    epochs: int = 10
    weight_decay: float = 1e-4
    threshold: float = 0.5
    validation_fraction: float = 0.1
    seed: int = 42


class FraudMLP(nn.Module):
    """Small feed-forward network used by centralized and later FL runs."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_client_training_data(data_dir: str | Path) -> tuple[pd.DataFrame, list[Path]]:
    data_dir = Path(data_dir)
    client_paths = sorted(data_dir.glob("client_*.csv"))
    if not client_paths:
        raise FileNotFoundError(f"No client_*.csv files found in {data_dir}.")
    frames = [pd.read_csv(path) for path in client_paths]
    for path, frame in zip(client_paths, frames, strict=True):
        _validate_frame(frame, path)
    return pd.concat(frames, ignore_index=True), client_paths


def _validate_frame(frame: pd.DataFrame, source: str | Path) -> None:
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns in {source}.")
    if frame.empty or frame[EXPECTED_COLUMNS].isna().any().any():
        raise ValueError(f"Empty data or missing values in {source}.")
    if not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"Non-binary labels in {source}.")


def to_tensors(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.from_numpy(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    labels = torch.from_numpy(frame[TARGET_COLUMN].to_numpy(dtype=np.float32))
    return features, labels


def train_mlp(
    training_frame: pd.DataFrame,
    config: MLPConfig,
    device: torch.device,
) -> tuple[FraudMLP, list[dict[str, float]]]:
    set_reproducible_seed(config.seed)
    features, labels = to_tensors(training_frame)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = FraudMLP(len(FEATURE_COLUMNS), config.hidden_sizes, config.dropout).to(device)
    fraud_count = float(labels.sum().item())
    normal_count = float(len(labels) - fraud_count)
    if fraud_count == 0 or normal_count == 0:
        raise ValueError("Training data must contain both normal and fraud examples.")
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(normal_count / fraud_count, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_labels)
        epoch_loss = total_loss / len(training_frame)
        history.append({"epoch": epoch, "training_loss": epoch_loss})
        print(f"Epoch {epoch:02d}/{config.epochs}: loss={epoch_loss:.6f}")
    return model, history


def evaluate_mlp(
    model: FraudMLP,
    test_frame: pd.DataFrame,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    _, labels = to_tensors(test_frame)
    scores = predict_probabilities(model, test_frame, batch_size, device)
    truth = labels.numpy().astype(np.int64)
    predictions = (scores >= threshold).astype(np.int64)
    return {
        "test_rows": len(truth),
        "test_fraud_count": int(truth.sum()),
        "threshold": threshold,
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
        "predicted_fraud_count": int(predictions.sum()),
    }


def predict_probabilities(
    model: FraudMLP,
    frame: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    features, _ = to_tensors(frame)
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_features,) in loader:
            probabilities.append(
                torch.sigmoid(model(batch_features.to(device))).cpu().numpy()
            )
    return np.concatenate(probabilities)


def select_f1_threshold(
    model: FraudMLP,
    validation_frame: pd.DataFrame,
    config: MLPConfig,
    device: torch.device,
) -> tuple[float, float]:
    """Select a cutoff using training-derived validation data, never global test data."""
    scores = predict_probabilities(model, validation_frame, config.batch_size, device)
    truth = validation_frame[TARGET_COLUMN].to_numpy(dtype=np.int64)
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    if len(thresholds) == 0:
        return config.threshold, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def write_metrics_report(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Week 2 Centralized MLP Results",
        "",
        "The model-development data came from the union of all six client-training",
        "partitions, with 10% reserved for threshold selection. The shared preprocessor",
        "was fitted only on client-training rows; `global_test.csv` was used only once",
        "for final evaluation.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Test rows | {metrics['test_rows']:,} |",
        f"| Test fraud cases | {metrics['test_fraud_count']:,} |",
        f"| Decision threshold | {metrics['threshold']:.2f} |",
        f"| Accuracy | {metrics['accuracy']:.6f} |",
        f"| Precision | {metrics['precision']:.6f} |",
        f"| Recall | {metrics['recall']:.6f} |",
        f"| F1 | {metrics['f1']:.6f} |",
        f"| AUC-ROC | {metrics['roc_auc']:.6f} |",
        f"| Average precision | {metrics['average_precision']:.6f} |",
        f"| Predicted fraud cases | {metrics['predicted_fraud_count']:,} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MLPConfig(
        epochs=args.epochs,
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

    training_raw, client_paths = load_client_training_data(args.data_dir)
    test_raw = pd.read_csv(args.data_dir / "global_test.csv")
    _validate_frame(test_raw, args.data_dir / "global_test.csv")
    preprocessor = FraudPreprocessor().fit(training_raw)
    training = preprocessor.transform(training_raw)
    test = preprocessor.transform(test_raw)

    model_training, validation = train_test_split(
        training,
        test_size=config.validation_fraction,
        random_state=config.seed,
        stratify=training[TARGET_COLUMN],
    )
    model, history = train_mlp(model_training.reset_index(drop=True), config, device)
    selected_threshold, validation_f1 = select_f1_threshold(
        model, validation.reset_index(drop=True), config, device
    )
    metrics = evaluate_mlp(model, test, selected_threshold, config.batch_size, device)
    run_metadata = {
        "model": "FraudMLP",
        "input_features": FEATURE_COLUMNS,
        "config": asdict(config),
        "device": str(device),
        "training_rows": len(training),
        "model_fit_rows": len(model_training),
        "validation_rows": len(validation),
        "training_fraud_count": int(training[TARGET_COLUMN].sum()),
        "threshold_selection": {
            "method": "maximum F1 on stratified client-training validation split",
            "selected_threshold": selected_threshold,
            "validation_f1": validation_f1,
        },
        "client_files": [path.name for path in client_paths],
        "history": history,
        "metrics": metrics,
    }

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(args.artifacts_dir / "preprocessor.json")
    torch.save(model.state_dict(), args.artifacts_dir / "centralized_mlp.pt")
    (args.artifacts_dir / "centralized_mlp_run.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    write_metrics_report(metrics, args.reports_dir / "week2_centralized_metrics.md")
    print(json.dumps(metrics, indent=2))
    print(f"Saved Week 2 artifacts to {args.artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
