"""Dataset validation, summarization, and non-IID partitioning utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"
EXPECTED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


def load_and_validate(path: str | Path) -> pd.DataFrame:
    """Load the dataset and fail early when its schema or values are invalid."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place Kaggle's creditcard.csv there first."
        )

    frame = pd.read_csv(path)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    extra_columns = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))
    if missing_columns or extra_columns:
        raise ValueError(f"Unexpected schema. Missing={missing_columns}; extra={extra_columns}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame[EXPECTED_COLUMNS].isna().any().any():
        missing = frame[EXPECTED_COLUMNS].isna().sum()
        raise ValueError(f"Dataset contains missing values: {missing[missing > 0].to_dict()}")
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in EXPECTED_COLUMNS):
        raise ValueError("All expected columns must be numeric.")

    labels = set(frame[TARGET_COLUMN].unique().tolist())
    if labels != {0, 1}:
        raise ValueError(f"Class must contain both binary labels 0 and 1; found {labels}.")
    return frame[EXPECTED_COLUMNS].copy()


def summarize_dataset(frame: pd.DataFrame) -> dict:
    """Return JSON-serializable Week 1 quality and imbalance statistics."""
    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    correlations = (
        frame.corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    return {
        "rows": len(frame),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "class_counts": {str(int(label)): int(count) for label, count in counts.items()},
        "fraud_rate": float(frame[TARGET_COLUMN].mean()),
        "top_absolute_target_correlations": {
            column: float(value) for column, value in correlations.head(10).items()
        },
        "amount_quantiles": {
            str(quantile): float(value)
            for quantile, value in frame["Amount"].quantile([0, 0.5, 0.9, 0.99, 1]).items()
        },
    }


def split_global_test(
    frame: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one untouched stratified test set before simulating clients."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame[TARGET_COLUMN],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def dirichlet_partition(
    frame: pd.DataFrame,
    num_clients: int = 6,
    alpha: float = 0.5,
    seed: int = 42,
    min_fraud_per_client: int = 10,
    max_attempts: int = 1_000,
) -> list[pd.DataFrame]:
    """Partition every row once using class-wise Dirichlet proportions.

    Sampling proportions independently for each class produces realistic variation
    in both client volume and fraud prevalence. A minimum positive count prevents
    unusable clients in this exceptionally imbalanced dataset.
    """
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    fraud_count = int(frame[TARGET_COLUMN].sum())
    if fraud_count < num_clients * min_fraud_per_client:
        raise ValueError(
            f"Need at least {num_clients * min_fraud_per_client} fraud rows, "
            f"but training data has {fraud_count}."
        )

    rng = np.random.default_rng(seed)
    labels = sorted(frame[TARGET_COLUMN].unique())
    for _ in range(max_attempts):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for label in labels:
            indices = frame.index[frame[TARGET_COLUMN] == label].to_numpy().copy()
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            counts = rng.multinomial(len(indices), proportions)
            boundaries = np.cumsum(counts)[:-1]
            for client_id, chunk in enumerate(np.split(indices, boundaries)):
                client_indices[client_id].extend(chunk.tolist())

        partitions = [
            frame.loc[indices].sample(frac=1, random_state=seed + client_id).reset_index(drop=True)
            for client_id, indices in enumerate(client_indices)
        ]
        if all(
            len(partition) > 0 and int(partition[TARGET_COLUMN].sum()) >= min_fraud_per_client
            for partition in partitions
        ):
            _validate_partition_integrity(frame, client_indices)
            return partitions

    raise RuntimeError(
        "Could not meet the minimum fraud constraint. Increase alpha, lower "
        "min_fraud_per_client, or reduce num_clients."
    )


def _validate_partition_integrity(frame: pd.DataFrame, indices: list[list[int]]) -> None:
    flattened = [index for client in indices for index in client]
    if len(flattened) != len(frame) or len(set(flattened)) != len(frame):
        raise RuntimeError("Partition integrity failure: rows were lost or duplicated.")
    if set(flattened) != set(frame.index):
        raise RuntimeError("Partition integrity failure: row indices do not match source data.")


def partition_manifest(partitions: list[pd.DataFrame]) -> pd.DataFrame:
    """Describe client volume and class skew for review and later experiments."""
    records = []
    for client_id, partition in enumerate(partitions):
        fraud_count = int(partition[TARGET_COLUMN].sum())
        records.append(
            {
                "client_id": client_id,
                "rows": len(partition),
                "normal_count": len(partition) - fraud_count,
                "fraud_count": fraud_count,
                "fraud_rate": float(partition[TARGET_COLUMN].mean()),
            }
        )
    return pd.DataFrame(records)


def save_week1_outputs(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    num_clients: int = 6,
    alpha: float = 0.5,
    test_size: float = 0.2,
    seed: int = 42,
    min_fraud_per_client: int = 10,
) -> pd.DataFrame:
    """Create and persist every Week 1 data artifact."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = split_global_test(frame, test_size=test_size, seed=seed)
    partitions = dirichlet_partition(
        train,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        min_fraud_per_client=min_fraud_per_client,
    )
    test.to_csv(output_dir / "global_test.csv", index=False)
    for client_id, partition in enumerate(partitions):
        partition.to_csv(output_dir / f"client_{client_id:02d}.csv", index=False)

    manifest = partition_manifest(partitions)
    manifest.to_csv(output_dir / "partition_manifest.csv", index=False)
    summary = summarize_dataset(frame)
    summary["partition_config"] = {
        "num_clients": num_clients,
        "dirichlet_alpha": alpha,
        "test_size": test_size,
        "seed": seed,
        "min_fraud_per_client": min_fraud_per_client,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return manifest
