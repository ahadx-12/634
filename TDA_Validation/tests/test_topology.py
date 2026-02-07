import numpy as np

from src.topology import TopologyExtractor


def test_entropy_empty():
    topo = TopologyExtractor()
    assert topo.compute_persistence_entropy(np.empty((0, 2))) == 0.0


def test_betti_counts_basic():
    # random small point cloud; we only assert it returns dict with keys
    rng = np.random.default_rng(0)
    pc = rng.normal(size=(80, 3))
    topo = TopologyExtractor(max_dimension=1, persistence_threshold=0.01)
    dgms = topo.compute_persistence(pc)
    betti = topo.extract_betti_numbers(dgms)
    assert 0 in betti and 1 in betti
    assert isinstance(betti[0], int)
