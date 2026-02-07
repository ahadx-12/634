import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.metrics import mutual_info_score


class TakensEmbedding:
    """Implements Takens' delay embedding theorem.

    Given a time series x(t), create vectors:
      X(i) = [x(i), x(i+τ), x(i+2τ), ..., x(i+(m-1)τ)]

    Parameters
    ---------
    delay : int | None
        Time delay τ. If None, computed using Average Mutual Information (AMI).
    dimension : int
        Embedding dimension m.
    n_bins : int
        Number of bins for discretization when computing AMI.
    """

    def __init__(self, delay=None, dimension=3, n_bins=30):
        self.delay = delay
        self.dimension = int(dimension)
        self.n_bins = int(n_bins)

    def fit(self, time_series):
        x = self._validate_series(time_series)
        if self.delay is None:
            self.delay = self._compute_optimal_delay(x)
        return self

    def transform(self, time_series):
        x = self._validate_series(time_series)
        if self.delay is None:
            raise ValueError("delay is None; call fit() first or pass delay explicitly")

        delay = int(self.delay)
        dim = int(self.dimension)
        if delay <= 0:
            raise ValueError("delay must be >= 1")
        if dim <= 1:
            raise ValueError("dimension must be >= 2")

        n = len(x)
        n_rows = n - (dim - 1) * delay
        if n_rows <= 0:
            raise ValueError(
                f"time_series too short for embedding: len={n}, dim={dim}, delay={delay}"
            )

        # Efficient view using stride tricks.
        # shape: (n_rows, dim)
        # strides: move 1 step in original series for next row; delay steps for next column
        shape = (n_rows, dim)
        strides = (x.strides[0], delay * x.strides[0])
        embedded = as_strided(x, shape=shape, strides=strides).copy()
        return embedded

    def _compute_optimal_delay(self, time_series, max_delay=50):
        x = self._validate_series(time_series)

        # For financial daily series, scanning to 30 is usually enough.
        max_delay = int(min(max_delay, max(5, len(x) // 10)))
        if max_delay < 2:
            return 1

        ami_values = []
        x_binned = self._bin_series(x, self.n_bins)

        for tau in range(1, max_delay + 1):
            a = x_binned[:-tau]
            b = x_binned[tau:]
            ami_values.append(mutual_info_score(a, b))

        ami_values = np.asarray(ami_values)

        # Find first local minimum: derivative changes from negative to positive.
        d = np.diff(ami_values)
        for i in range(1, len(d)):
            if d[i - 1] < 0 and d[i] > 0:
                return i + 1  # because tau index starts at 1

        # Fallback: choose tau at global minimum
        return int(np.argmin(ami_values) + 1)

    @staticmethod
    def _validate_series(time_series):
        x = np.asarray(time_series, dtype=float)
        if x.ndim != 1:
            raise ValueError("time_series must be 1D")
        if len(x) < 10:
            raise ValueError("time_series too short")
        if not np.isfinite(x).all():
            raise ValueError("time_series contains NaN/inf")
        return x

    @staticmethod
    def _bin_series(x, n_bins):
        # Robust binning: if constant series, all bins same
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if np.isclose(xmin, xmax):
            return np.zeros_like(x, dtype=int)
        bins = np.linspace(xmin, xmax, n_bins + 1)
        binned = np.digitize(x, bins[1:-1], right=False)
        return binned
