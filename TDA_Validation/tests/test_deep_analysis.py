"""Deep analysis of the TDA pipeline behavior.

Focuses on:
1. False positive root cause on stable data
2. Behavior on realistic financial-like data (GBM / random walk)
3. Sensitivity vs specificity tradeoff
4. Whether topology genuinely adds value over simpler statistics
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embedding import TakensEmbedding
from src.topology import TopologyExtractor
from src.detector import TopologicalAnomalyDetector
from src.multi_scale_detector import MultiScaleDetector


def make_gbm(n=500, mu=0.0005, sigma=0.02, s0=100):
    """Geometric Brownian Motion (realistic stock prices)."""
    returns = np.random.normal(mu, sigma, n)
    prices = s0 * np.exp(np.cumsum(returns))
    return prices


def make_gbm_with_crash(n_normal=400, n_crash=20, n_post=100, crash_magnitude=-0.08):
    """GBM with an embedded crash event."""
    normal = make_gbm(n_normal, mu=0.0005, sigma=0.015)
    # Crash: series of large negative returns
    crash_returns = np.random.normal(crash_magnitude, 0.03, n_crash)
    crash_prices = normal[-1] * np.exp(np.cumsum(crash_returns))
    # Post-crash: higher volatility recovery
    post_returns = np.random.normal(0.001, 0.04, n_post)
    post_prices = crash_prices[-1] * np.exp(np.cumsum(post_returns))
    return np.concatenate([normal, crash_prices, post_prices])


print("=" * 60)
print("DEEP ANALYSIS OF TDA PIPELINE")
print("=" * 60)


# ============================================================
# 1. FALSE POSITIVE ROOT CAUSE ANALYSIS
# ============================================================
print("\n--- 1. FALSE POSITIVE ROOT CAUSE ANALYSIS ---")

np.random.seed(42)
# Test: pure sine (the failing test case)
x = np.sin(2 * np.pi * 0.05 * np.arange(600)) + 0.02 * np.random.randn(600)
det = TopologicalAnomalyDetector(window_size=50, baseline_period=6, sensitivity=1.5)
det.fit(x[:300])

print(f"  Adaptive thresholds: {det.adaptive_thresholds}")
print(f"  Baseline betti: {det.baseline_betti}")
print(f"  Baseline entropy: {det.baseline_entropy:.4f}")

# Check what signals are triggering
signal_counts = {"betti": 0, "wasserstein": 0, "entropy": 0}
anomaly_count = 0
for i in range(300, 600):
    if i < 50:
        continue
    w = x[i - 50:i]
    res = det.detect(w)
    if res["is_anomaly"]:
        anomaly_count += 1
    for sig in ["betti", "wasserstein", "entropy"]:
        if res["signals"][sig]:
            signal_counts[sig] += 1

total_tests = 300
print(f"  Anomalies: {anomaly_count}/{total_tests} ({anomaly_count/total_tests:.1%})")
print(f"  Signal firing rates:")
for sig, cnt in signal_counts.items():
    print(f"    {sig}: {cnt}/{total_tests} ({cnt/total_tests:.1%})")

# Diagnosis: which 2-of-3 combinations trigger most?
print("\n  Root cause: Since 2-of-3 consensus is needed, the FP rate depends on")
print("  how correlated the three signals are. On a stable series with tiny variation,")
print("  the adaptive thresholds become very tight, making the detector oversensitive.")


# ============================================================
# 2. BEHAVIOR ON REALISTIC FINANCIAL DATA (GBM)
# ============================================================
print("\n--- 2. GBM (REALISTIC STOCK PRICES) ---")

for trial in range(3):
    np.random.seed(trial * 100)
    prices = make_gbm(800, mu=0.0003, sigma=0.015)

    msd = MultiScaleDetector(scales=[30, 50, 80], min_scales=2, sensitivity=0.75, baseline_period=4)
    msd.fit(prices[:400])

    anomalies = 0
    tests = 0
    for i in range(400, len(prices)):
        if i < 240:
            continue
        hist = prices[max(0, i - 240):i]
        res = msd.detect(hist)
        if res["is_anomaly"]:
            anomalies += 1
        tests += 1

    fp_rate = anomalies / max(1, tests)
    print(f"  Trial {trial}: {anomalies}/{tests} alerts ({fp_rate:.1%})")


# ============================================================
# 3. CRASH DETECTION ON GBM + CRASH
# ============================================================
print("\n--- 3. CRASH DETECTION ON GBM ---")

detected_crashes = 0
lead_times = []
n_trials = 5

for trial in range(n_trials):
    np.random.seed(trial * 50 + 7)
    prices = make_gbm_with_crash(n_normal=400, n_crash=20, n_post=100, crash_magnitude=-0.06)
    crash_start = 400

    msd = MultiScaleDetector(scales=[30, 50, 80], min_scales=2, sensitivity=0.75, baseline_period=4)
    msd.fit(prices[:300])

    # Check if anything triggers in the 10 days before crash
    pre_crash_detected = False
    best_lead = 0
    for i in range(max(240, 300), crash_start + 15):
        if i >= len(prices):
            break
        hist = prices[max(0, i - 240):i]
        res = msd.detect(hist)
        days_before_crash = crash_start - i
        if res["is_anomaly"] and -2 <= days_before_crash <= 10:
            pre_crash_detected = True
            best_lead = max(best_lead, days_before_crash)

    if pre_crash_detected:
        detected_crashes += 1
        lead_times.append(best_lead)

    status = "DETECTED" if pre_crash_detected else "MISSED"
    print(f"  Trial {trial}: {status} (lead={best_lead}d)")

det_rate = detected_crashes / n_trials
avg_lead = np.mean(lead_times) if lead_times else 0
print(f"  Detection rate: {detected_crashes}/{n_trials} ({det_rate:.0%})")
print(f"  Avg lead time: {avg_lead:.1f} days")


# ============================================================
# 4. DOES TOPOLOGY ADD VALUE OVER SIMPLE VOLATILITY?
# ============================================================
print("\n--- 4. TOPOLOGY vs SIMPLE VOLATILITY COMPARISON ---")
print("  Comparing TDA detector vs simple rolling volatility z-score")

np.random.seed(42)
prices = make_gbm_with_crash(n_normal=400, n_crash=20, n_post=100, crash_magnitude=-0.07)
crash_start = 400

# Method A: TDA detector
msd = MultiScaleDetector(scales=[30, 50, 80], min_scales=2, sensitivity=0.75, baseline_period=4)
msd.fit(prices[:300])

tda_pre_crash = []
for i in range(300, min(crash_start + 20, len(prices))):
    if i < 240:
        continue
    hist = prices[max(0, i - 240):i]
    res = msd.detect(hist)
    days_to_crash = crash_start - i
    if res["is_anomaly"] and -5 <= days_to_crash <= 15:
        tda_pre_crash.append(days_to_crash)

# Method B: Simple rolling volatility z-score
returns = np.diff(np.log(prices))
vol_window = 20
baseline_vol = np.std(returns[:300])
baseline_mean_vol = np.mean([np.std(returns[i:i+vol_window]) for i in range(0, 280, 5)])

vol_pre_crash = []
for i in range(300, min(crash_start + 20, len(prices) - 1)):
    recent_vol = np.std(returns[max(0, i - vol_window):i])
    vol_zscore = (recent_vol - baseline_mean_vol) / (baseline_vol + 1e-9)
    days_to_crash = crash_start - i
    if vol_zscore > 2.0 and -5 <= days_to_crash <= 15:
        vol_pre_crash.append(days_to_crash)

print(f"  TDA alerts within [-5, +15] days of crash: {len(tda_pre_crash)}")
if tda_pre_crash:
    print(f"    Earliest: {max(tda_pre_crash)} days before crash")
print(f"  Volatility alerts within [-5, +15] days of crash: {len(vol_pre_crash)}")
if vol_pre_crash:
    print(f"    Earliest: {max(vol_pre_crash)} days before crash")

if tda_pre_crash and (not vol_pre_crash or max(tda_pre_crash) > max(vol_pre_crash)):
    print("  >> TDA provides earlier warning than simple volatility")
elif vol_pre_crash and (not tda_pre_crash or max(vol_pre_crash) > max(tda_pre_crash)):
    print("  >> Simple volatility provides earlier warning than TDA")
else:
    print("  >> Both methods comparable (or neither detected)")


# ============================================================
# 5. SENSITIVITY SWEEP
# ============================================================
print("\n--- 5. SENSITIVITY PARAMETER SWEEP ---")

np.random.seed(42)
stable = make_gbm(600, mu=0.0003, sigma=0.015)
crash_series = make_gbm_with_crash(400, 20, 100, crash_magnitude=-0.07)

for sens in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    # FP rate on stable
    msd = MultiScaleDetector(scales=[30, 50, 80], min_scales=2, sensitivity=sens, baseline_period=4)
    msd.fit(stable[:300])
    fp = 0
    fp_total = 0
    for i in range(300, 600):
        if i < 240:
            continue
        hist = stable[max(0, i - 240):i]
        if msd.detect(hist)["is_anomaly"]:
            fp += 1
        fp_total += 1
    fp_rate = fp / max(1, fp_total)

    # Detection on crash
    msd2 = MultiScaleDetector(scales=[30, 50, 80], min_scales=2, sensitivity=sens, baseline_period=4)
    msd2.fit(crash_series[:300])
    detected = False
    for i in range(350, 420):
        if i >= len(crash_series) or i < 240:
            continue
        hist = crash_series[max(0, i - 240):i]
        if msd2.detect(hist)["is_anomaly"]:
            detected = True
            break

    print(f"  sens={sens:>4.2f}: FP rate={fp_rate:>5.1%}, crash detected={detected}")


# ============================================================
# 6. INFORMATION CONTENT OF INDIVIDUAL METRICS
# ============================================================
print("\n--- 6. METRIC DISCRIMINATIVE POWER ---")

np.random.seed(42)
stable_data = make_gbm(500, sigma=0.015)
crash_data = make_gbm_with_crash(400, 20, 80, crash_magnitude=-0.07)

det = TopologicalAnomalyDetector(window_size=50, baseline_period=4, sensitivity=1.0)
det.fit(stable_data[:200])

# Collect metrics from stable windows
stable_metrics = {"betti": [], "wasserstein": [], "entropy": []}
for i in range(200, 400):
    if i < 50:
        continue
    w = stable_data[i-50:i]
    res = det.detect(w)
    stable_metrics["betti"].append(res["metrics"]["betti_score"])
    stable_metrics["wasserstein"].append(res["metrics"]["wasserstein"])
    stable_metrics["entropy"].append(res["metrics"]["entropy_delta"])

det2 = TopologicalAnomalyDetector(window_size=50, baseline_period=4, sensitivity=1.0)
det2.fit(crash_data[:200])

# Collect metrics from crash windows
crash_metrics = {"betti": [], "wasserstein": [], "entropy": []}
for i in range(390, min(430, len(crash_data))):
    if i < 50:
        continue
    w = crash_data[i-50:i]
    res = det2.detect(w)
    crash_metrics["betti"].append(res["metrics"]["betti_score"])
    crash_metrics["wasserstein"].append(res["metrics"]["wasserstein"])
    crash_metrics["entropy"].append(res["metrics"]["entropy_delta"])

for metric in ["betti", "wasserstein", "entropy"]:
    s_mean = np.mean(stable_metrics[metric])
    s_std = np.std(stable_metrics[metric])
    c_mean = np.mean(crash_metrics[metric])
    c_std = np.std(crash_metrics[metric])
    # Cohen's d effect size
    pooled_std = np.sqrt((s_std**2 + c_std**2) / 2)
    cohens_d = (c_mean - s_mean) / (pooled_std + 1e-9)
    print(f"  {metric:>12}: stable={s_mean:.4f}+/-{s_std:.4f}  crash={c_mean:.4f}+/-{c_std:.4f}  Cohen's d={cohens_d:.2f}")


print("\n" + "=" * 60)
print("DEEP ANALYSIS COMPLETE")
print("=" * 60)
