import numpy as np

from .embedding import TakensEmbedding
from .topology import TopologyExtractor


class TopologicalAnomalyDetector:
    """Detect anomalies by comparing topology of a window to a baseline.

    Refinement notes:
    - Uses multi-metric consensus voting (2-out-of-3).
    - Uses adaptive thresholds learned from baseline variation:
        threshold = mean + sensitivity * std
    """

    def __init__(self, window_size=100, baseline_period=30, sensitivity=2.0, zscore_normalize=True):
        self.window_size = int(window_size)
        self.baseline_period = int(baseline_period)
        self.sensitivity = float(sensitivity)
        self.zscore_normalize = bool(zscore_normalize)

        self.embedding = TakensEmbedding(dimension=3)
        self.topology = TopologyExtractor(max_dimension=2, persistence_threshold=0.1)

        self.baseline_betti = None
        self.baseline_diagrams = None  # representative H1 diagram
        self.baseline_entropy = None
        self.baseline_windows = []

        # Adaptive thresholds learned in fit()
        self.adaptive_thresholds = None  # {betti_score, wasserstein, entropy}

        # Betti change significance for the *component-level* change dict (kept as a fallback)
        self._betti_threshold = 2

    def fit(self, time_series):
        """Learn baseline topology AND adaptive thresholds.

        We build `baseline_period` overlapping windows (50% overlap) from the
        start of the series, compute topology for each, then measure *natural
        variation* between consecutive baseline windows.

        Thresholds:
          threshold = mean(variation) + sensitivity * std(variation)
        """

        x = np.asarray(time_series, dtype=float)
        if x.ndim == 2 and 1 in x.shape:
            x = x.reshape(-1)
        if len(x) < 2 * self.window_size:
            raise ValueError("Not enough data to fit baseline")

        step = max(1, self.window_size // 2)  # 50% overlap
        self.baseline_windows = []
        for i in range(self.baseline_period):
            start = i * step
            end = start + self.window_size
            if end > len(x):
                break
            self.baseline_windows.append(x[start:end])

        if len(self.baseline_windows) < 2:
            raise ValueError("Not enough data to fit baseline windows")

        # Fit embedding delay once on baseline data for stability
        self.embedding.fit(self._prep_series(self.baseline_windows[0]))

        baseline_topologies = []
        for w in self.baseline_windows:
            w = self._prep_series(w)
            pc = self.embedding.transform(w)
            dgms = self.topology.compute_persistence(pc)
            betti = self.topology.extract_betti_numbers(dgms)
            h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
            # Use combined entropy (H0 + H1) as a more sensitive complexity measure.
            h0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
            ent = self.topology.compute_persistence_entropy(h0) + self.topology.compute_persistence_entropy(h1)
            baseline_topologies.append({"betti": betti, "h1": h1, "entropy": ent})

        # Baseline = median Betti, median entropy
        b0 = [t["betti"].get(0, 0) for t in baseline_topologies]
        b1 = [t["betti"].get(1, 0) for t in baseline_topologies]
        b2 = [t["betti"].get(2, 0) for t in baseline_topologies]
        self.baseline_betti = {0: int(np.median(b0)), 1: int(np.median(b1)), 2: int(np.median(b2))}
        self.baseline_entropy = float(np.median([t["entropy"] for t in baseline_topologies]))

        # Representative baseline H1 diagram:
        # Use a medoid that minimizes total Wasserstein distance to other baseline windows.
        h1_list = [t["h1"] for t in baseline_topologies]
        if len(h1_list) == 1:
            self.baseline_diagrams = h1_list[0]
        else:
            totals = []
            for i in range(len(h1_list)):
                s = 0.0
                for j in range(len(h1_list)):
                    if i == j:
                        continue
                    s += self.topology.wasserstein_distance(h1_list[i], h1_list[j])
                totals.append(s)
            rep_idx = int(np.argmin(np.asarray(totals)))
            self.baseline_diagrams = h1_list[rep_idx]

        # Measure natural variation around the baseline.
        # This is more aligned with how detection works (current vs baseline)
        # than comparing consecutive baseline windows.
        betti_scores = []
        wasserstein_vars = []
        entropy_vars = []

        for t in baseline_topologies:
            diff0 = abs(t["betti"].get(0, 0) - self.baseline_betti.get(0, 0))
            diff1 = abs(t["betti"].get(1, 0) - self.baseline_betti.get(1, 0))
            diff2 = abs(t["betti"].get(2, 0) - self.baseline_betti.get(2, 0))
            # Use an unweighted sum to avoid shrinking the metric.
            betti_scores.append(diff0 + diff1 + diff2)

            wasserstein_vars.append(self.topology.wasserstein_distance(t["h1"], self.baseline_diagrams))
            entropy_vars.append(abs(t["entropy"] - self.baseline_entropy))

        def _thr(vals, fallback):
            vals = np.asarray(vals, dtype=float)
            if len(vals) == 0:
                return float(fallback)
            return float(np.mean(vals) + self.sensitivity * np.std(vals))

        self.adaptive_thresholds = {
            "betti_score": _thr(betti_scores, 1.0),
            "wasserstein": _thr(wasserstein_vars, 0.5),
            "entropy": _thr(entropy_vars, 0.3),
        }

        return self

    def detect(self, new_window):
        if self.baseline_betti is None or self.baseline_diagrams is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        w = self._prep_series(np.asarray(new_window, dtype=float))
        pc = self.embedding.transform(w)
        dgms = self.topology.compute_persistence(pc)
        betti_current = self.topology.extract_betti_numbers(dgms)

        h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
        h0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
        entropy_current = self.topology.compute_persistence_entropy(h0) + self.topology.compute_persistence_entropy(h1)
        dist = self.topology.wasserstein_distance(h1, self.baseline_diagrams)

        betti_change = {
            0: int(abs(betti_current.get(0, 0) - self.baseline_betti.get(0, 0))),
            1: int(abs(betti_current.get(1, 0) - self.baseline_betti.get(1, 0))),
            2: int(abs(betti_current.get(2, 0) - self.baseline_betti.get(2, 0))),
        }

        if self.adaptive_thresholds is None:
            raise ValueError("Adaptive thresholds not computed. Call fit() first.")

        # --- Three independent signals ---
        # Metric 1: Betti score vs adaptive threshold
        betti_score = betti_change[0] + betti_change[1] + betti_change[2]
        betti_flag = betti_score > self.adaptive_thresholds["betti_score"]

        # Metric 2: Wasserstein distance vs adaptive threshold
        dist_flag = dist > self.adaptive_thresholds["wasserstein"]

        # Metric 3: entropy delta vs adaptive threshold
        entropy_delta = abs(entropy_current - self.baseline_entropy)
        entropy_flag = entropy_delta > self.adaptive_thresholds["entropy"]

        # --- Consensus voting (2-out-of-3) ---
        # Rationale: reduces false positives by requiring agreement among metrics.
        signals = [betti_flag, dist_flag, entropy_flag]
        num_agreeing = int(sum(bool(s) for s in signals))
        is_anomaly = num_agreeing >= 2
        confidence = num_agreeing / 3.0

        # score in [0,1] via soft normalization against thresholds
        score = 0.0
        score += 0.4 * min(1.0, dist / (self.adaptive_thresholds["wasserstein"] + 1e-9))
        score += 0.3 * min(1.0, entropy_delta / (self.adaptive_thresholds["entropy"] + 1e-9))
        score += 0.3 * min(1.0, betti_score / (self.adaptive_thresholds["betti_score"] + 1e-9))

        anomaly_type = self._classify_anomaly_type(betti_current, self.baseline_betti) if is_anomaly else "NORMAL"

        return {
            "is_anomaly": bool(is_anomaly),
            "confidence": float(confidence),
            "score": float(score),
            "type": anomaly_type,
            # Topology snapshots
            "betti_current": betti_current,
            "betti_baseline": self.baseline_betti,
            "betti_change": betti_change,
            "wasserstein_dist": float(dist),
            "entropy_current": float(entropy_current),
            "entropy_baseline": float(self.baseline_entropy),
            "entropy_change": float(entropy_delta),
            # Signal debug
            "signals": {
                "betti": bool(betti_flag),
                "wasserstein": bool(dist_flag),
                "entropy": bool(entropy_flag),
                "num_agreeing": int(num_agreeing),
            },
            "thresholds": dict(self.adaptive_thresholds),
            "metrics": {
                "betti_score": float(betti_score),
                "wasserstein": float(dist),
                "entropy_delta": float(entropy_delta),
            },
        }

    def _classify_anomaly_type(self, betti_current, betti_baseline):
        b0c, b1c, b2c = betti_current.get(0, 0), betti_current.get(1, 0), betti_current.get(2, 0)
        b0b, b1b, b2b = betti_baseline.get(0, 0), betti_baseline.get(1, 0), betti_baseline.get(2, 0)

        if b0c > b0b:
            return "fragmentation (β0 increased)"
        if b2c > b2b:
            return "void formation (β2 increased)"
        if b1c > b1b:
            return "bifurcation/complexity (β1 increased)"
        if b1c == 0 and b1b > 0:
            return "collapse (β1 dropped to 0)"
        return "deviation"

    def _prep_series(self, x):
        x = np.asarray(x, dtype=float)
        # Be tolerant to column-vectors from pandas/numpy (shape (N,1) or (1,N)).
        if x.ndim == 2 and 1 in x.shape:
            x = x.reshape(-1)
        if x.ndim != 1:
            raise ValueError("series/window must be 1D")
        if len(x) != self.window_size:
            # allow longer series by taking last window_size
            if len(x) < self.window_size:
                raise ValueError("window too short")
            x = x[-self.window_size :]

        if self.zscore_normalize:
            mu = float(np.mean(x))
            sigma = float(np.std(x))
            if sigma < 1e-12:
                return x - mu
            return (x - mu) / sigma
        return x

    @staticmethod
    def _sliding_windows(x, window_size):
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n < window_size:
            return []
        return [x[i - window_size : i] for i in range(window_size, n + 1)]
