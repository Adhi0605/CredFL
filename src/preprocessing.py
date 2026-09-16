"""Leakage-safe feature preprocessing for centralized and federated training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.data import FEATURE_COLUMNS, TARGET_COLUMN

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
TimeStrategy = Literal["standard", "cyclical"]


@dataclass(frozen=True)
class PreprocessingState:
    """Serializable parameters fitted exclusively on training data."""

    time_strategy: TimeStrategy
    amount_transform: str
    amount_mean: float
    amount_scale: float
    time_mean: float | None
    time_scale: float | None
    input_features: list[str]
    output_features: list[str]


class FraudPreprocessor:
    """Transform fraud features consistently without changing PCA components."""

    def __init__(self, time_strategy: TimeStrategy = "standard") -> None:
        if time_strategy not in ("standard", "cyclical"):
            raise ValueError("time_strategy must be 'standard' or 'cyclical'.")
        self.time_strategy = time_strategy
        self.state: PreprocessingState | None = None

    def fit(self, training_frame: pd.DataFrame) -> FraudPreprocessor:
        """Fit scaling statistics using training rows only."""
        _validate_features(training_frame)
        amount = training_frame["Amount"].to_numpy(dtype=np.float64)
        if np.any(amount < 0):
            raise ValueError("Amount must be non-negative before applying log1p.")

        log_amount = np.log1p(amount)
        amount_mean, amount_scale = _mean_and_scale(log_amount)

        time_mean: float | None = None
        time_scale: float | None = None
        if self.time_strategy == "standard":
            time = training_frame["Time"].to_numpy(dtype=np.float64)
            time_mean, time_scale = _mean_and_scale(time)
            output_features = FEATURE_COLUMNS.copy()
        else:
            output_features = ["time_sin", "time_cos", *PCA_COLUMNS, "Amount"]

        self.state = PreprocessingState(
            time_strategy=self.time_strategy,
            amount_transform="log1p_then_standard",
            amount_mean=amount_mean,
            amount_scale=amount_scale,
            time_mean=time_mean,
            time_scale=time_scale,
            input_features=FEATURE_COLUMNS.copy(),
            output_features=output_features,
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted training statistics while preserving an optional target."""
        if self.state is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        _validate_features(frame)

        result = pd.DataFrame(index=frame.index)
        time = frame["Time"].to_numpy(dtype=np.float64)
        if self.state.time_strategy == "standard":
            result["Time"] = (time - self.state.time_mean) / self.state.time_scale
        else:
            seconds_in_day = np.mod(time, 86_400.0)
            angle = 2.0 * np.pi * seconds_in_day / 86_400.0
            result["time_sin"] = np.sin(angle)
            result["time_cos"] = np.cos(angle)

        for column in PCA_COLUMNS:
            result[column] = frame[column].to_numpy(copy=True)

        amount = frame["Amount"].to_numpy(dtype=np.float64)
        if np.any(amount < 0):
            raise ValueError("Amount must be non-negative before applying log1p.")
        result["Amount"] = (
            np.log1p(amount) - self.state.amount_mean
        ) / self.state.amount_scale

        if TARGET_COLUMN in frame.columns:
            result[TARGET_COLUMN] = frame[TARGET_COLUMN].to_numpy(copy=True)
        return result.reset_index(drop=True)

    def fit_transform(self, training_frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(training_frame).transform(training_frame)

    def save(self, path: str | Path) -> None:
        """Save transform choices, fitted parameters, and feature ordering."""
        if self.state is None:
            raise RuntimeError("Preprocessor must be fitted before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> FraudPreprocessor:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        preprocessor = cls(time_strategy=values["time_strategy"])
        preprocessor.state = PreprocessingState(**values)
        return preprocessor


def _validate_features(frame: pd.DataFrame) -> None:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if frame.empty:
        raise ValueError("Cannot preprocess an empty dataset.")
    if frame[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature columns cannot contain missing values.")
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in FEATURE_COLUMNS):
        raise ValueError("All feature columns must be numeric.")


def _mean_and_scale(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    scale = float(values.std(ddof=0))
    if scale == 0.0:
        scale = 1.0
    return mean, scale
