"""Multi-scale topological anomaly detection.

Day 3: Multi-Timeframe Confirmation

Idea:
- Run the same topological anomaly detection at multiple window sizes.
- Require anomalies to be confirmed at >= min_scales time scales.

This reduces scale-specific noise (false positives) while preserving true
regime-change signals that persist across scales.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .detector import TopologicalAnomalyDetector


class MultiScaleDetector:
    """Detect anomalies using multiple time scales (window sizes)."""

    def __init__(
        self,
        scales=(30, 50, 100),
        min_scales=2,
        sensitivity=2.0,
        baseline_period=None,
    ):
        self.scales = list(scales)
        self.min_scales = int(min_scales)
        self.sensitivity = float(sensitivity)

        # If baseline_period is provided, use it for all scales.
        # Otherwise, use a scale-dependent default.
        self._baseline_period_override = baseline_period

        self.detectors = {
            int(scale): TopologicalAnomalyDetector(
                window_size=int(scale),
                baseline_period=self._baseline_period_for_scale(int(scale)),
                sensitivity=self.sensitivity,
            )
            for scale in self.scales
        }

    def set_baseline_period(self, baseline_period: int | None):
        """Override baseline_period for all scales (rebuilds detectors)."""
        self._baseline_period_override = baseline_period
        self.detectors = {
            int(scale): TopologicalAnomalyDetector(
                window_size=int(scale),
                baseline_period=self._baseline_period_for_scale(int(scale)),
                sensitivity=self.sensitivity,
            )
            for scale in self.scales
        }
        return self

    def _baseline_period_for_scale(self, scale: int) -> int:
        if self._baseline_period_override is not None:
            return int(self._baseline_period_override)
        # Spec default was max(10, scale//5). We keep that intent but allow fit()
        # to downsample if not enough windows are available.
        return max(10, scale // 5)

    def set_sensitivity(self, sensitivity: float):
        self.sensitivity = float(sensitivity)
        for det in self.detectors.values():
            det.sensitivity = float(sensitivity)
        return self

    def fit(self, time_series):
        x = np.asarray(time_series, dtype=float)
        if x.ndim == 2 and 1 in x.shape:
            x = x.reshape(-1)

        # Minimum requirement: enough data for at least 2 windows at the largest scale.
        min_required = max(self.scales) * 2
        if len(x) < min_required:
            raise ValueError(
                f"Need at least {min_required} data points for multi-scale fit; got {len(x)}"
            )

        for scale, det in self.detectors.items():
            # Fit each detector on the full history; the detector will build its
            # own baseline windows from the start.
            det.fit(x)
        return self

    def detect(self, time_series):
        x = np.asarray(time_series, dtype=float)
        if x.ndim == 2 and 1 in x.shape:
            x = x.reshape(-1)

        if len(x) < max(self.scales):
            raise ValueError(
                f"Time series too short. Need >= {max(self.scales)} points, got {len(x)}"
            )

        scale_results = {}
        agreeing_scales = []

        for scale in self.scales:
            det = self.detectors[int(scale)]
            window = x[-int(scale) :]
            res = det.detect(window)
            scale_results[int(scale)] = res
            if res.get("is_anomaly"):
                agreeing_scales.append(int(scale))

        num_detecting = len(agreeing_scales)
        is_anomaly = num_detecting >= self.min_scales
        confidence = num_detecting / max(1, len(self.scales))

        if is_anomaly:
            types = [scale_results[s]["type"] for s in agreeing_scales]
            anomaly_type = Counter(types).most_common(1)[0][0]
        else:
            anomaly_type = "NORMAL"

        return {
            "is_anomaly": bool(is_anomaly),
            "confidence": float(confidence),
            "type": anomaly_type,
            "num_detecting": int(num_detecting),
            "agreeing_scales": agreeing_scales,
            "scale_results": scale_results,
        }

    def detect_streaming(self, new_point, history):
        hist = np.asarray(history, dtype=float)
        if hist.ndim == 2 and 1 in hist.shape:
            hist = hist.reshape(-1)
        series = np.append(hist, float(new_point))
        series = series[-max(self.scales) * 3 :]
        return self.detect(series)
