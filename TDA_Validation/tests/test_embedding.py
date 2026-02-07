import numpy as np

from src.embedding import TakensEmbedding


def test_embedding_shape():
    ts = np.random.randn(100)
    emb = TakensEmbedding(delay=2, dimension=3)
    result = emb.fit(ts).transform(ts)
    assert result.shape == (96, 3)


def test_embedding_values():
    ts = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    emb = TakensEmbedding(delay=1, dimension=3)
    # allow short series for this test
    emb._validate_series = staticmethod(lambda x: np.asarray(x, dtype=float))
    result = emb.fit(ts).transform(ts)
    assert np.allclose(result[0], [1, 2, 3])
    assert np.allclose(result[1], [2, 3, 4])


def test_optimal_delay_sine_wave():
    t = np.linspace(0, 100, 1000)
    ts = np.sin(2 * np.pi * t / 20)
    emb = TakensEmbedding(delay=None, dimension=3)
    emb.fit(ts)
    assert 3 <= emb.delay <= 10
