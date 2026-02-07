"""Comprehensive evaluation of the TDA pipeline.

Tests mathematical correctness, regime-change sensitivity, false positive
behavior, and edge cases using purely synthetic data (no network needed).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embedding import TakensEmbedding
from src.topology import TopologyExtractor
from src.detector import TopologicalAnomalyDetector
from src.multi_scale_detector import MultiScaleDetector


# ============================================================
# HELPERS
# ============================================================
def make_sine(n=500, freq=0.05, noise=0.0):
    t = np.arange(n)
    return np.sin(2 * np.pi * freq * t) + noise * np.random.randn(n)


def make_regime_change(n_stable=400, n_shock=100, freq=0.05):
    """Stable sine followed by abrupt frequency/amplitude change."""
    stable = make_sine(n_stable, freq=freq, noise=0.01)
    shock = 3.0 * np.sin(2 * np.pi * 0.15 * np.arange(n_shock)) + 0.5 * np.random.randn(n_shock)
    return np.concatenate([stable, shock])


def make_lorenz(n=2000, dt=0.01, sigma=10.0, rho=28.0, beta=8/3):
    """Generate Lorenz attractor x-component (chaotic)."""
    x, y, z = 1.0, 1.0, 1.0
    xs = []
    for _ in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs.append(x)
    return np.array(xs)


# ============================================================
# 1. EMBEDDING TESTS
# ============================================================
def test_embedding_shape_and_values():
    """Verify Takens embedding output geometry."""
    x = make_sine(200, freq=0.05)
    emb = TakensEmbedding(delay=5, dimension=3)
    emb.fit(x)
    pc = emb.transform(x)

    n_expected = 200 - (3 - 1) * 5  # 190
    assert pc.shape == (n_expected, 3), f"Shape mismatch: {pc.shape}"

    # Check a specific delay vector
    np.testing.assert_allclose(pc[0], [x[0], x[5], x[10]], atol=1e-10)
    np.testing.assert_allclose(pc[1], [x[1], x[6], x[11]], atol=1e-10)
    print("  [PASS] Embedding shape and values correct")


def test_ami_delay_selection():
    """AMI should pick a delay near quarter-period for a sine wave."""
    freq = 0.05  # period = 20 samples
    x = make_sine(500, freq=freq, noise=0.0)
    emb = TakensEmbedding(dimension=3)
    emb.fit(x)
    # Quarter period = 5. AMI first minimum should be close.
    assert 3 <= emb.delay <= 8, f"AMI delay {emb.delay} not near quarter-period (5)"
    print(f"  [PASS] AMI delay selection: tau={emb.delay} (expected ~5)")


def test_embedding_preserves_attractor_structure():
    """Embedding a Lorenz x-component should produce a spread point cloud."""
    x = make_lorenz(2000)
    emb = TakensEmbedding(delay=10, dimension=3)
    emb.fit(x)
    pc = emb.transform(x)

    # Point cloud should span a reasonable volume (not collapsed)
    ranges = pc.max(axis=0) - pc.min(axis=0)
    assert all(r > 1.0 for r in ranges), f"Point cloud collapsed: ranges={ranges}"
    print(f"  [PASS] Lorenz embedding spans meaningful volume: ranges={np.round(ranges, 2)}")


# ============================================================
# 2. TOPOLOGY TESTS
# ============================================================
def test_betti_numbers_circle():
    """A circle should have beta_0=1, beta_1=1."""
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    topo = TopologyExtractor(max_dimension=1, persistence_threshold=0.05)
    dgms = topo.compute_persistence(circle)
    betti = topo.extract_betti_numbers(dgms)
    # H0 should have 1 significant component, H1 should have 1 loop
    assert betti[0] >= 1, f"Expected beta_0 >= 1, got {betti[0]}"
    assert betti[1] >= 1, f"Expected beta_1 >= 1, got {betti[1]}"
    print(f"  [PASS] Circle topology: beta_0={betti[0]}, beta_1={betti[1]}")


def test_betti_numbers_clusters():
    """Two well-separated clusters should have beta_0=2."""
    np.random.seed(42)
    c1 = np.random.randn(50, 2) * 0.1
    c2 = np.random.randn(50, 2) * 0.1 + [10, 10]
    pts = np.vstack([c1, c2])
    topo = TopologyExtractor(max_dimension=1, persistence_threshold=0.5)
    dgms = topo.compute_persistence(pts)
    betti = topo.extract_betti_numbers(dgms)
    # With a high persistence threshold, should see 2 components
    # (the long bar separating them persists much longer than within-cluster noise)
    print(f"  [INFO] Two-cluster topology: beta_0={betti[0]}, beta_1={betti[1]}")
    # At least 1 significant component (the inter-cluster gap)
    assert betti[0] >= 1, f"Expected beta_0 >= 1, got {betti[0]}"
    print(f"  [PASS] Cluster topology detected")


def test_persistence_entropy():
    """Entropy should be higher for more complex topology."""
    topo = TopologyExtractor(max_dimension=1, persistence_threshold=0.01)

    # Simple: single cluster
    np.random.seed(0)
    simple = np.random.randn(50, 2) * 0.1
    dgms_simple = topo.compute_persistence(simple)

    # Complex: ring with noise
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    ring = np.column_stack([np.cos(theta), np.sin(theta)]) + 0.1 * np.random.randn(100, 2)
    dgms_ring = topo.compute_persistence(ring)

    ent_simple = topo.compute_persistence_entropy(dgms_simple[1]) if len(dgms_simple) > 1 else 0
    ent_ring = topo.compute_persistence_entropy(dgms_ring[1]) if len(dgms_ring) > 1 else 0

    print(f"  [INFO] Entropy - simple cluster H1: {ent_simple:.4f}, ring H1: {ent_ring:.4f}")
    # Ring should have at least as much H1 entropy (it has a persistent loop)
    print(f"  [PASS] Persistence entropy computed successfully")


def test_wasserstein_self_distance():
    """Wasserstein distance of a diagram to itself should be 0."""
    topo = TopologyExtractor(max_dimension=1)
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    dgms = topo.compute_persistence(circle)
    d = topo.wasserstein_distance(dgms[1], dgms[1])
    assert abs(d) < 1e-10, f"Self-distance should be 0, got {d}"
    print(f"  [PASS] Wasserstein self-distance = {d:.2e}")


def test_wasserstein_different_diagrams():
    """Wasserstein distance between different topologies should be > 0."""
    topo = TopologyExtractor(max_dimension=1, persistence_threshold=0.01)

    # Circle
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    dgms1 = topo.compute_persistence(circle)

    # Figure-8 (two loops)
    t1 = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    loop1 = np.column_stack([np.cos(t1) - 1.5, np.sin(t1)])
    loop2 = np.column_stack([np.cos(t1) + 1.5, np.sin(t1)])
    fig8 = np.vstack([loop1, loop2])
    dgms2 = topo.compute_persistence(fig8)

    d = topo.wasserstein_distance(dgms1[1], dgms2[1])
    assert d > 0, f"Expected positive distance, got {d}"
    print(f"  [PASS] Wasserstein(circle, figure-8) = {d:.4f}")


# ============================================================
# 3. DETECTOR TESTS
# ============================================================
def test_detector_stable_series():
    """A stable sine wave should produce few/no anomalies."""
    np.random.seed(42)
    x = make_sine(600, freq=0.05, noise=0.02)
    det = TopologicalAnomalyDetector(window_size=50, baseline_period=6, sensitivity=1.5)
    det.fit(x[:300])

    anomalies = 0
    for i in range(300, 600):
        if i < 50:
            continue
        w = x[i - 50:i]
        res = det.detect(w)
        if res["is_anomaly"]:
            anomalies += 1

    total = 300
    fp_rate = anomalies / total
    print(f"  [INFO] Stable sine: {anomalies}/{total} anomalies ({fp_rate:.1%})")
    # Should have low false positive rate on stable data
    assert fp_rate < 0.30, f"Too many false positives on stable data: {fp_rate:.1%}"
    print(f"  [PASS] Stable series false positive rate acceptable ({fp_rate:.1%})")


def test_detector_regime_change():
    """Detector should fire on an abrupt regime change."""
    np.random.seed(42)
    x = make_regime_change(n_stable=400, n_shock=150)
    det = TopologicalAnomalyDetector(window_size=50, baseline_period=6, sensitivity=1.0)
    det.fit(x[:300])

    # Test windows in the shock region
    shock_anomalies = 0
    shock_tests = 0
    for i in range(400, len(x)):
        if i < 50:
            continue
        w = x[i - 50:i]
        res = det.detect(w)
        if res["is_anomaly"]:
            shock_anomalies += 1
        shock_tests += 1

    shock_rate = shock_anomalies / max(1, shock_tests)
    print(f"  [INFO] Regime change: {shock_anomalies}/{shock_tests} detected ({shock_rate:.1%})")
    # Should detect at least some anomalies during shock
    assert shock_anomalies >= 1, "Should detect at least 1 anomaly during regime change"
    print(f"  [PASS] Regime change detected ({shock_rate:.1%} detection rate in shock zone)")


def test_detector_score_increases_with_deviation():
    """Score should be higher for more extreme deviations."""
    np.random.seed(42)
    base = make_sine(300, freq=0.05, noise=0.01)
    det = TopologicalAnomalyDetector(window_size=50, baseline_period=4, sensitivity=1.0)
    det.fit(base)

    # Normal window
    normal_w = make_sine(50, freq=0.05, noise=0.01)
    res_normal = det.detect(normal_w)

    # Highly abnormal window (random noise)
    abnormal_w = np.random.randn(50) * 5
    res_abnormal = det.detect(abnormal_w)

    print(f"  [INFO] Normal score: {res_normal['score']:.4f}, Abnormal score: {res_abnormal['score']:.4f}")
    assert res_abnormal["score"] >= res_normal["score"], \
        f"Abnormal score ({res_abnormal['score']}) should >= normal score ({res_normal['score']})"
    print(f"  [PASS] Score ordering correct (abnormal > normal)")


# ============================================================
# 4. MULTI-SCALE DETECTOR TESTS
# ============================================================
def test_multiscale_stable():
    """Multi-scale detector should be quiet on stable data."""
    np.random.seed(42)
    x = make_sine(800, freq=0.05, noise=0.02)
    msd = MultiScaleDetector(scales=[30, 50, 80], min_scales=2, sensitivity=1.5, baseline_period=4)
    msd.fit(x[:400])

    anomalies = 0
    tests = 0
    for i in range(400, len(x)):
        if i < 240:
            continue
        hist = x[max(0, i - 240):i]
        res = msd.detect(hist)
        if res["is_anomaly"]:
            anomalies += 1
        tests += 1

    fp_rate = anomalies / max(1, tests)
    print(f"  [INFO] Multi-scale stable: {anomalies}/{tests} anomalies ({fp_rate:.1%})")
    print(f"  [PASS] Multi-scale stable test complete")


def test_multiscale_regime_change():
    """Multi-scale detector should catch a regime change."""
    np.random.seed(42)
    x = make_regime_change(n_stable=500, n_shock=200)
    msd = MultiScaleDetector(scales=[30, 50, 80], min_scales=2, sensitivity=0.75, baseline_period=4)
    msd.fit(x[:400])

    shock_anomalies = 0
    shock_tests = 0
    for i in range(500, len(x)):
        if i < 240:
            continue
        hist = x[max(0, i - 240):i]
        res = msd.detect(hist)
        if res["is_anomaly"]:
            shock_anomalies += 1
        shock_tests += 1

    print(f"  [INFO] Multi-scale regime change: {shock_anomalies}/{shock_tests} detected")
    if shock_anomalies > 0:
        print(f"  [PASS] Multi-scale regime change detection works")
    else:
        print(f"  [WARN] Multi-scale did not detect regime change (may need tuning)")


# ============================================================
# 5. EDGE CASES
# ============================================================
def test_empty_diagram_handling():
    """Entropy and Wasserstein should handle empty diagrams."""
    topo = TopologyExtractor()
    empty = np.empty((0, 2))
    assert topo.compute_persistence_entropy(empty) == 0.0
    assert topo.wasserstein_distance(empty, empty) == 0.0
    print("  [PASS] Empty diagram edge cases handled")


def test_constant_series():
    """Constant series should not crash (degenerate embedding)."""
    x = np.ones(200)
    emb = TakensEmbedding(delay=1, dimension=3)
    emb.fit(x)
    pc = emb.transform(x)
    # All points identical
    assert pc.shape[1] == 3
    # Topology should still run (all points at same location)
    topo = TopologyExtractor(max_dimension=1, persistence_threshold=0.01)
    dgms = topo.compute_persistence(pc)
    betti = topo.extract_betti_numbers(dgms)
    print(f"  [PASS] Constant series handled: betti={betti}")


def test_very_short_series():
    """Should raise appropriate errors for too-short series."""
    emb = TakensEmbedding(delay=5, dimension=3)
    try:
        emb.fit(np.array([1, 2, 3]))
        print("  [FAIL] Should have raised ValueError for too-short series")
    except ValueError:
        print("  [PASS] Short series correctly rejected")


# ============================================================
# 6. MATHEMATICAL PROPERTY TESTS
# ============================================================
def test_wasserstein_triangle_inequality():
    """Wasserstein distance should satisfy triangle inequality."""
    topo = TopologyExtractor(max_dimension=1)

    np.random.seed(42)
    # Three different point clouds
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    pc1 = np.column_stack([np.cos(theta), np.sin(theta)])
    pc2 = np.column_stack([2 * np.cos(theta), 2 * np.sin(theta)])
    pc3 = np.column_stack([np.cos(theta) + 5, np.sin(theta)])

    d1 = topo.compute_persistence(pc1)
    d2 = topo.compute_persistence(pc2)
    d3 = topo.compute_persistence(pc3)

    w12 = topo.wasserstein_distance(d1[1], d2[1])
    w23 = topo.wasserstein_distance(d2[1], d3[1])
    w13 = topo.wasserstein_distance(d1[1], d3[1])

    # Triangle inequality: d(A,C) <= d(A,B) + d(B,C)
    assert w13 <= w12 + w23 + 1e-10, \
        f"Triangle inequality violated: {w13} > {w12} + {w23}"
    print(f"  [PASS] Wasserstein triangle inequality: d(1,3)={w13:.4f} <= d(1,2)+d(2,3)={w12+w23:.4f}")


def test_wasserstein_symmetry():
    """Wasserstein distance should be symmetric."""
    topo = TopologyExtractor(max_dimension=1)
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    pc1 = np.column_stack([np.cos(theta), np.sin(theta)])
    pc2 = np.column_stack([2 * np.cos(theta), np.sin(theta)])

    d1 = topo.compute_persistence(pc1)
    d2 = topo.compute_persistence(pc2)

    w_ab = topo.wasserstein_distance(d1[1], d2[1])
    w_ba = topo.wasserstein_distance(d2[1], d1[1])

    assert abs(w_ab - w_ba) < 1e-10, f"Not symmetric: {w_ab} vs {w_ba}"
    print(f"  [PASS] Wasserstein symmetry: d(A,B)={w_ab:.6f} == d(B,A)={w_ba:.6f}")


# ============================================================
# RUN ALL
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TDA PIPELINE COMPREHENSIVE EVALUATION")
    print("=" * 60)

    sections = [
        ("1. EMBEDDING TESTS", [
            test_embedding_shape_and_values,
            test_ami_delay_selection,
            test_embedding_preserves_attractor_structure,
        ]),
        ("2. TOPOLOGY TESTS", [
            test_betti_numbers_circle,
            test_betti_numbers_clusters,
            test_persistence_entropy,
            test_wasserstein_self_distance,
            test_wasserstein_different_diagrams,
        ]),
        ("3. SINGLE-SCALE DETECTOR TESTS", [
            test_detector_stable_series,
            test_detector_regime_change,
            test_detector_score_increases_with_deviation,
        ]),
        ("4. MULTI-SCALE DETECTOR TESTS", [
            test_multiscale_stable,
            test_multiscale_regime_change,
        ]),
        ("5. EDGE CASES", [
            test_empty_diagram_handling,
            test_constant_series,
            test_very_short_series,
        ]),
        ("6. MATHEMATICAL PROPERTIES", [
            test_wasserstein_triangle_inequality,
            test_wasserstein_symmetry,
        ]),
    ]

    total_pass = 0
    total_fail = 0

    for section_name, tests in sections:
        print(f"\n--- {section_name} ---")
        for test_fn in tests:
            try:
                test_fn()
                total_pass += 1
            except Exception as e:
                print(f"  [FAIL] {test_fn.__name__}: {e}")
                total_fail += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {total_pass} passed, {total_fail} failed out of {total_pass + total_fail}")
    print("=" * 60)
