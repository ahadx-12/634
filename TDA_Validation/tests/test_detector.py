import numpy as np

from src.detector import TopologicalAnomalyDetector


def test_fit_and_detect_runs():
    rng = np.random.default_rng(0)
    ts = rng.normal(size=400).cumsum()  # random walk-ish

    det = TopologicalAnomalyDetector(window_size=80, baseline_period=10)
    det.fit(ts)

    window = ts[-80:]
    res = det.detect(window)

    assert "is_anomaly" in res
    assert "score" in res
    assert 0.0 <= res["score"] <= 1.2  # allow slight >1 due to rounding
