"""Refinement pipeline helpers (Day 4-7).

This module keeps validator.py from becoming too large.
All outputs written under results/.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from .multi_scale_detector import MultiScaleDetector


def ensure_results_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/reports", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)


def expanded_events():
    return [
        # High priority
        {"date": "2020-02-24", "name": "COVID Crash Start", "ticker": "SPY", "priority": "HIGH"},
        {"date": "2020-03-12", "name": "COVID Black Thursday", "ticker": "SPY", "priority": "HIGH"},
        {"date": "2018-02-05", "name": "Volmageddon", "ticker": "SPY", "priority": "HIGH"},
        {"date": "2015-08-24", "name": "China Flash Crash", "ticker": "SPY", "priority": "HIGH"},
        {"date": "2011-08-08", "name": "US Downgrade", "ticker": "SPY", "priority": "HIGH"},
        {"date": "2010-05-06", "name": "Flash Crash", "ticker": "SPY", "priority": "HIGH"},
        # Medium priority
        {"date": "2022-01-03", "name": "Fed Pivot to QT", "ticker": "SPY", "priority": "MEDIUM"},
        {"date": "2018-10-10", "name": "Powell Pivot", "ticker": "SPY", "priority": "MEDIUM"},
        {"date": "2016-06-24", "name": "Brexit Vote", "ticker": "SPY", "priority": "MEDIUM"},
        {"date": "2013-05-22", "name": "Taper Tantrum", "ticker": "SPY", "priority": "MEDIUM"},
        # Low priority
        {"date": "2023-03-10", "name": "SVB Collapse", "ticker": "XLF", "priority": "LOW"},
        {"date": "2020-04-20", "name": "Oil Negative", "ticker": "USO", "priority": "LOW"},
    ]


def download_with_retry(ticker, start, end, max_retries=3):
    last_err = None
    for _ in range(max_retries):
        try:
            d = yf.download(ticker, start=start, end=end, progress=False)
            if d is not None and not d.empty:
                return d
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to download {ticker}: {last_err}")


def analyze_false_positives_detailed(detector: MultiScaleDetector, year="2019"):
    """Deep false positive analysis on a calm year."""
    ensure_results_dirs()

    start = f"{year}-01-01"
    end = f"{year}-12-31"

    data = download_with_retry("SPY", start, end)
    vix = download_with_retry("^VIX", start, end)

    prices = data["Close"].values
    dates = data.index
    vix_values = vix["Close"].reindex(dates, method="ffill").values

    fit_n = min(400, len(prices) - 1)
    if fit_n < 300:
        return {"total_fps": 0, "fp_rate": 0.0, "details": []}

    det = MultiScaleDetector(
        scales=detector.scales,
        min_scales=detector.min_scales,
        sensitivity=detector.sensitivity,
        baseline_period=4,
    )
    det.fit(prices[:fit_n])

    fps = []
    total = 0
    for i in range(fit_n, len(prices)):
        if i < max(det.scales):
            continue
        hist = prices[max(0, i - max(det.scales) * 3) : i]
        res = det.detect(hist)
        if res["is_anomaly"]:
            fps.append(
                {
                    "date": str(dates[i].date()),
                    "vix": float(vix_values[i]) if i < len(vix_values) else None,
                    "confidence": float(res.get("confidence", 0.0)),
                    "agreeing_scales": list(res.get("agreeing_scales", [])),
                }
            )
        total += 1

    fp_rate = len(fps) / max(1, total)
    return {"total_fps": len(fps), "fp_rate": float(fp_rate), "details": fps}


def parameter_grid_search():
    """Day 5 quick grid search.

    Goal: maximize F1 with constraints: high_priority >=80%, fp<=20%.
    Uses a reduced evaluation set for speed.
    """

    param_grid = {
        "sensitivity": [0.5, 0.75, 1.0, 1.25, 1.5],
        "min_scales": [2],
        "scales": [[15, 40, 80], [20, 40, 80], [20, 50, 80], [20, 50, 100]],
    }

    test_events = [
        ("2020-02-24", "SPY"),
        ("2018-02-05", "SPY"),
        ("2015-08-24", "SPY"),
        ("2011-08-08", "SPY"),
        ("2010-05-06", "SPY"),
    ]

    results = []

    for sens in param_grid["sensitivity"]:
        for min_sc in param_grid["min_scales"]:
            for scales in param_grid["scales"]:
                det = MultiScaleDetector(scales=scales, min_scales=min_sc, sensitivity=sens, baseline_period=4)

                detected = 0
                leads = []

                for date, ticker in test_events:
                    event_dt = pd.to_datetime(date)
                    d = download_with_retry(ticker, event_dt - pd.Timedelta(days=1200), event_dt + pd.Timedelta(days=5))
                    prices = d["Close"].values
                    dates = d.index
                    train_n = min(400, len(prices) - 1)
                    if train_n < 300 or len(prices) < (train_n + max(scales) + 10):
                        continue
                    det.fit(prices[:train_n])

                    ok = False
                    lead = 0
                    for i in range(train_n, len(prices)):
                        if i < max(scales):
                            continue
                        hist = prices[max(0, i - max(scales) * 3) : i]
                        res = det.detect(hist)
                        days_before = int((event_dt - dates[i]).days)
                        if res["is_anomaly"] and 0 <= days_before <= 7:
                            ok = True
                            lead = max(lead, days_before)
                    if ok:
                        detected += 1
                        leads.append(lead)

                det_rate = detected / max(1, len(test_events))
                avg_lead = float(np.mean(leads)) if leads else 0.0

                fp = analyze_false_positives_detailed(det, year="2019")
                fp_rate = fp["fp_rate"]

                precision = det_rate / max(1e-9, (det_rate + fp_rate))
                recall = det_rate
                f1 = (2 * precision * recall / max(1e-9, (precision + recall)))

                results.append(
                    {
                        "scales": scales,
                        "min_scales": min_sc,
                        "sensitivity": sens,
                        "det_rate": det_rate,
                        "fp_rate": fp_rate,
                        "avg_lead": avg_lead,
                        "f1": f1,
                    }
                )

    # Filter and pick best
    valid = [r for r in results if r["det_rate"] >= 0.7 and r["fp_rate"] <= 0.2]
    if valid:
        best = max(valid, key=lambda r: r["f1"])
    else:
        best = max(results, key=lambda r: r["f1"]) if results else None

    return best, results


def save_optimal_config(best):
    ensure_results_dirs()
    if not best:
        return None
    cfg = {
        "scales": best["scales"],
        "min_scales": best["min_scales"],
        "sensitivity": best["sensitivity"],
        "performance": {
            "det_rate": best["det_rate"],
            "fp_rate": best["fp_rate"],
            "avg_lead": best["avg_lead"],
            "f1": best["f1"],
        },
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    with open("results/optimal_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg
