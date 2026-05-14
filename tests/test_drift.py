import numpy as np
import pytest
from app.drift import DriftDetector

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    cur = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur, alpha=0.05)
    assert result["drift_detected"] is False
    assert result["n_drifted_features"] == 0


def test_drift_on_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    cur = rng.normal(loc=8.0, scale=1.0, size=(500, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur, alpha=0.05)
    assert result["drift_detected"] is True
    assert result["n_drifted_features"] == 4
    for feat in FEATURE_NAMES:
        assert result["per_feature"][feat]["p_value"] < 0.05


def test_detect_returns_correct_structure():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(100, 4))
    cur = rng.normal(size=(50, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur)
    assert "drift_detected" in result
    assert "n_drifted_features" in result
    assert "drifted_features" in result
    assert "per_feature" in result
    assert result["n_samples"] == 50
    assert result["alpha"] == 0.05
    for feat in FEATURE_NAMES:
        assert feat in result["per_feature"]
        info = result["per_feature"][feat]
        assert "statistic" in info
        assert "p_value" in info
        assert "drift_detected" in info


def test_invalid_reference_raises():
    with pytest.raises(ValueError):
        DriftDetector(np.array([1, 2, 3]), FEATURE_NAMES)


def test_invalid_current_raises():
    ref = np.random.normal(size=(100, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    with pytest.raises(ValueError):
        detector.detect(np.random.normal(size=(50, 3)))


def test_custom_alpha():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    cur = rng.normal(loc=5.05, scale=1.0, size=(500, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    result_strict = detector.detect(cur, alpha=0.001)
    result_loose = detector.detect(cur, alpha=0.5)
    assert result_strict["alpha"] == 0.001
    assert result_loose["alpha"] == 0.5
