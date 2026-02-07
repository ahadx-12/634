import numpy as np
from ripser import ripser
from scipy.optimize import linear_sum_assignment


class TopologyExtractor:
    """Compute persistent homology features using Vietoris-Rips complex.

    Note: We implement a lightweight p-Wasserstein distance between persistence
    diagrams ourselves to avoid the `persim` dependency (which pulls in
    matplotlib on Windows and can cause file-lock install errors).
    """

    def __init__(self, max_dimension=2, persistence_threshold=0.1):
        self.max_dimension = int(max_dimension)
        self.persistence_threshold = float(persistence_threshold)

    def compute_persistence(self, point_cloud):
        x = np.asarray(point_cloud, dtype=float)
        if x.ndim != 2:
            raise ValueError("point_cloud must be 2D")
        res = ripser(x, maxdim=self.max_dimension)
        return res["dgms"]

    def extract_betti_numbers(self, diagrams):
        betti = {}
        for k, dgm in enumerate(diagrams):
            if dgm.size == 0:
                betti[k] = 0
                continue

            dgm = np.asarray(dgm, dtype=float)
            finite = dgm[np.isfinite(dgm[:, 1])]
            if finite.size == 0:
                betti[k] = 0
                continue

            persistence = finite[:, 1] - finite[:, 0]
            significant = persistence > self.persistence_threshold
            betti[k] = int(np.sum(significant))
        return betti

    def compute_persistence_entropy(self, diagram):
        dgm = np.asarray(diagram, dtype=float)
        if dgm.size == 0:
            return 0.0

        finite = dgm[np.isfinite(dgm[:, 1])]
        if finite.size == 0:
            return 0.0

        p = finite[:, 1] - finite[:, 0]
        p = p[p > 0]
        if len(p) == 0:
            return 0.0

        p = p / np.sum(p)
        return float(-np.sum(p * np.log(p + 1e-12)))

    @staticmethod
    def _project_to_diagonal(points):
        # Projection of (b,d) to diagonal is ((b+d)/2, (b+d)/2)
        m = 0.5 * (points[:, 0] + points[:, 1])
        return np.stack([m, m], axis=1)

    def wasserstein_distance(self, diagram1, diagram2, order=2):
        """Compute p-Wasserstein distance between two persistence diagrams.

        We use an assignment formulation that allows matching points to the
        diagonal (i.e., deleting features).

        This is a practical implementation for small diagrams typical of our
        window sizes.
        """

        p = float(order)
        d1 = np.asarray(diagram1, dtype=float)
        d2 = np.asarray(diagram2, dtype=float)

        # Remove infinite deaths (can't be transported meaningfully in this simple metric)
        if d1.size == 0:
            d1 = d1.reshape(0, 2)
        if d2.size == 0:
            d2 = d2.reshape(0, 2)
        d1 = d1[np.isfinite(d1[:, 1])]
        d2 = d2[np.isfinite(d2[:, 1])]

        n, m = len(d1), len(d2)
        if n == 0 and m == 0:
            return 0.0

        # Cost matrix size (n+m) x (n+m)
        # Top-left: match d1 points to d2 points
        # Top-right: match d1 points to diagonal (deletion)
        # Bottom-left: match diagonal (insertion) to d2 points
        # Bottom-right: zeros (diag to diag)
        N = n + m
        C = np.zeros((N, N), dtype=float)

        if n > 0 and m > 0:
            diff = d1[:, None, :] - d2[None, :, :]
            dist = np.linalg.norm(diff, axis=2)  # L2
            C[:n, :m] = dist**p

        if n > 0:
            d1_diag = self._project_to_diagonal(d1)
            del_cost = np.linalg.norm(d1 - d1_diag, axis=1) ** p
            C[:n, m:] = del_cost[:, None]

        if m > 0:
            d2_diag = self._project_to_diagonal(d2)
            ins_cost = np.linalg.norm(d2 - d2_diag, axis=1) ** p
            C[n:, :m] = ins_cost[None, :]

        row_ind, col_ind = linear_sum_assignment(C)
        total = C[row_ind, col_ind].sum()
        return float(total ** (1.0 / p))
