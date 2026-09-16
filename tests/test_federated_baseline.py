import numpy as np
import pandas as pd
import pytest
import torch

from src.data import EXPECTED_COLUMNS
from src.federated_baseline import (
    FederatedConfig,
    split_client_frames,
    train_federated,
    weighted_fedavg,
)


def make_client(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(rows, len(EXPECTED_COLUMNS))), columns=EXPECTED_COLUMNS)
    frame["Amount"] = np.abs(frame["Amount"])
    frame["Class"] = np.tile([0, 1], rows // 2)
    frame["V1"] = frame["Class"] * 3.0 + rng.normal(0, 0.1, rows)
    return frame


def test_weighted_fedavg_uses_client_example_counts() -> None:
    states = [
        {"weight": torch.tensor([1.0, 3.0])},
        {"weight": torch.tensor([5.0, 7.0])},
    ]
    result = weighted_fedavg(states, [1, 3])
    torch.testing.assert_close(result["weight"], torch.tensor([4.0, 6.0]))


def test_weighted_fedavg_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        weighted_fedavg([], [])
    with pytest.raises(ValueError):
        weighted_fedavg([{"weight": torch.tensor([1.0])}], [0])


def test_client_split_preserves_rows_and_holds_out_each_client() -> None:
    clients = [make_client(40, 1), make_client(60, 2)]
    training, validation = split_client_frames(clients, validation_fraction=0.2, seed=42)
    assert [len(frame) for frame in training] == [32, 48]
    assert len(validation) == 20
    assert sum(len(frame) for frame in training) + len(validation) == 100
    assert all(frame["Class"].nunique() == 2 for frame in training)


def test_tiny_federated_run_returns_round_history() -> None:
    clients = [make_client(32, 1), make_client(48, 2)]
    config = FederatedConfig(
        hidden_sizes=(8,), dropout=0.0, rounds=2, local_epochs=1, batch_size=16
    )
    model, history = train_federated(clients, config, torch.device("cpu"))
    assert len(history) == 2
    assert model(torch.zeros((3, 30))).shape == (3,)
    assert all(np.isfinite(item["weighted_training_loss"]) for item in history)
    assert all(item["participating_clients"] == 2 for item in history)
