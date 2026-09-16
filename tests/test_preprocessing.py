import json

import numpy as np
import pandas as pd

from src.data import EXPECTED_COLUMNS
from src.preprocessing import PCA_COLUMNS, FraudPreprocessor


def make_frame() -> pd.DataFrame:
    rows = 4
    frame = pd.DataFrame(0.0, index=range(rows), columns=EXPECTED_COLUMNS)
    frame["Time"] = [0.0, 21_600.0, 43_200.0, 86_400.0]
    frame["Amount"] = [0.0, 9.0, 99.0, 999.0]
    frame["V1"] = [1.0, 2.0, 3.0, 4.0]
    frame["Class"] = [0, 0, 1, 0]
    return frame


def test_standard_preprocessing_log_scales_amount_and_preserves_pca_and_target() -> None:
    frame = make_frame()
    transformed = FraudPreprocessor().fit_transform(frame)

    assert np.isclose(transformed["Amount"].mean(), 0.0)
    assert np.isclose(transformed["Amount"].std(ddof=0), 1.0)
    assert np.isclose(transformed["Time"].mean(), 0.0)
    assert np.isclose(transformed["Time"].std(ddof=0), 1.0)
    assert transformed[PCA_COLUMNS].equals(frame[PCA_COLUMNS])
    assert transformed["Class"].equals(frame["Class"])


def test_cyclical_time_uses_seconds_within_day() -> None:
    transformed = FraudPreprocessor(time_strategy="cyclical").fit_transform(make_frame())

    assert "Time" not in transformed
    assert np.allclose(transformed["time_sin"], [0.0, 1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(transformed["time_cos"], [1.0, 0.0, -1.0, 1.0], atol=1e-12)


def test_saved_state_reproduces_transform(tmp_path) -> None:
    frame = make_frame()
    preprocessor = FraudPreprocessor().fit(frame.iloc[:3])
    state_path = tmp_path / "preprocessor.json"
    preprocessor.save(state_path)
    restored = FraudPreprocessor.load(state_path)

    pd.testing.assert_frame_equal(preprocessor.transform(frame), restored.transform(frame))
    metadata = json.loads(state_path.read_text(encoding="utf-8"))
    assert metadata["amount_transform"] == "log1p_then_standard"
    assert metadata["output_features"] == EXPECTED_COLUMNS[:-1]
