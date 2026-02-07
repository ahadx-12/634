import numpy as np

from src.multi_scale_detector import MultiScaleDetector


def test_multiscale_fit_detect_runs():
    rng = np.random.default_rng(0)
    prices = rng.normal(size=600).cumsum()

    ms = MultiScaleDetector(scales=[30, 50, 100], min_scales=2, sensitivity=1.0)
    ms.fit(prices)

    res = ms.detect(prices)
    assert "is_anomaly" in res
    assert "confidence" in res
    assert "scale_results" in res
    assert set(res["scale_results"].keys()) == {30, 50, 100}
