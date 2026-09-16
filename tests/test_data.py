import numpy as np
import pandas as pd

from src.data import (
    EXPECTED_COLUMNS,
    dirichlet_partition,
    partition_manifest,
    split_global_test,
    summarize_dataset,
)


def make_frame(normal: int = 900, fraud: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(normal + fraud, len(EXPECTED_COLUMNS) - 1))
    frame = pd.DataFrame(values, columns=EXPECTED_COLUMNS[:-1])
    frame["Amount"] = np.abs(frame["Amount"])
    frame["Class"] = [0] * normal + [1] * fraud
    return frame


def test_summary_reports_imbalance() -> None:
    summary = summarize_dataset(make_frame())
    assert summary["rows"] == 1_000
    assert summary["class_counts"] == {"0": 900, "1": 100}
    assert summary["fraud_rate"] == 0.1


def test_global_test_is_stratified_and_complete() -> None:
    frame = make_frame()
    train, test = split_global_test(frame, test_size=0.2, seed=42)
    assert len(train) == 800
    assert len(test) == 200
    assert int(train["Class"].sum()) == 80
    assert int(test["Class"].sum()) == 20


def test_dirichlet_partition_preserves_rows_and_minimum_fraud() -> None:
    frame = make_frame()
    partitions = dirichlet_partition(
        frame, num_clients=4, alpha=1.0, seed=42, min_fraud_per_client=5
    )
    assert sum(map(len, partitions)) == len(frame)
    assert sum(int(partition["Class"].sum()) for partition in partitions) == 100
    assert all(int(partition["Class"].sum()) >= 5 for partition in partitions)
    assert len(partition_manifest(partitions)) == 4
