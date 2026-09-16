import numpy as np
import pandas as pd
import torch

from src.centralized_baseline import FraudMLP, MLPConfig, evaluate_mlp, train_mlp
from src.data import EXPECTED_COLUMNS


def make_learnable_frame(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(rows, len(EXPECTED_COLUMNS))), columns=EXPECTED_COLUMNS)
    frame["Amount"] = np.abs(frame["Amount"])
    frame["Class"] = np.tile([0, 1], rows // 2)
    frame["V1"] = frame["Class"] * 3.0 + rng.normal(0, 0.1, rows)
    return frame


def test_mlp_has_one_logit_per_row() -> None:
    model = FraudMLP(input_size=30, hidden_sizes=(8, 4), dropout=0.0)
    assert model(torch.zeros((5, 30))).shape == (5,)


def test_training_and_evaluation_return_finite_metrics() -> None:
    frame = make_learnable_frame()
    config = MLPConfig(hidden_sizes=(8,), dropout=0.0, epochs=2, batch_size=16)
    model, history = train_mlp(frame, config, torch.device("cpu"))
    metrics = evaluate_mlp(model, frame, 0.5, 16, torch.device("cpu"))

    assert len(history) == 2
    assert all(np.isfinite(item["training_loss"]) for item in history)
    for name in ("precision", "recall", "f1", "roc_auc", "average_precision"):
        assert 0.0 <= metrics[name] <= 1.0
