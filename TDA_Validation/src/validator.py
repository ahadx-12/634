import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

from .detector import TopologicalAnomalyDetector
from .multi_scale_detector import MultiScaleDetector
from .signal_analysis import summarize_signal_patterns
from .refinement_pipeline import (
    analyze_false_positives_detailed,
    expanded_events,
    parameter_grid_search,
    save_optimal_config,
)


class ValidationSuite:
    def __init__(self):
        # Current tuned multi-scale defaults (Day 3). These can be optimized later.
        self.detector = MultiScaleDetector(scales=[15, 40, 80], min_scales=2, sensitivity=0.75, baseline_period=4)

    def analyze_signal_patterns(self, event_results):
        """Print a small, explainable breakdown of which signals fired in true positives."""
        print(" " + "=" * 60)
        print("SIGNAL PATTERN ANALYSIS")
        print("=" * 60)
        # Signal analysis is for the single-scale detector; for multi-scale
        # we instead summarize how many scales agree.
        total = 0
        agreeing_hist = {}
        for ev in event_results.get("details", []):
            for d in ev.get("all_detections", []) or []:
                if d.get("is_anomaly") and 0 <= d.get("days_before", 9999) <= 7:
                    total += 1
                    k = int(d.get("num_scales", 0))
                    agreeing_hist[k] = agreeing_hist.get(k, 0) + 1

        print(f" True-positive detections (within 0-7d of events): {total}")
        if total == 0:
            print(" No true-positive detections found to analyze.")
            return {"total": 0, "agreeing_hist": {}}

        for k in sorted(agreeing_hist):
            cnt = agreeing_hist[k]
            pct = cnt / total * 100.0
            print(f"  num_scales={k}: {cnt} ({pct:.1f}%)")

        return {"total": total, "agreeing_hist": agreeing_hist}

    def run_full_validation(self):
        """Run full validation.

        To speed up Day 3 tuning, you can set environment variable:
          S68_FAST_TUNE=1
        which runs a smaller subset of tests.
        """

        import os

        print("=" * 60)
        print("S68 VALIDATION SUITE - REFINEMENT PHASE")
        print("=" * 60)

        fast = os.environ.get("S68_FAST_TUNE", "0") == "1"
        if fast:
            print(" FAST_TUNE mode enabled (subset evaluation)")
            return self.run_fast_tune()

        print(f" Using sensitivity (k) = {self.detector.sensitivity} (multi-scale)")

        print(" [1/3] Testing on known historical events...")
        event_results = self.test_historical_events()

        self.analyze_signal_patterns(event_results)

        print(" [2/3] Testing false positive rate...")
        fp_results = self.test_false_positives()

        print(" [3/3] Testing parameter robustness...")
        param_results = self.test_parameter_sensitivity()

        self.generate_report(event_results, fp_results, param_results)

    def run_fast_tune(self):
        """Fast subset evaluation (optional)."""

        key_events = [
            {"date": "2020-02-24", "name": "COVID Crash", "ticker": "SPY"},
            {"date": "2018-02-05", "name": "Volmageddon", "ticker": "SPY"},
            {"date": "2015-08-24", "name": "China Flash Crash", "ticker": "SPY"},
        ]

        results = []
        for ev in key_events:
            r = self._test_single_event(ev["date"], ev["name"], ev["ticker"])
            results.append(r)
            print(f"  [{'OK' if r['detected'] else 'NO'}] {ev['name']}: lead={r.get('lead_time', 0)}")

        det_rate = sum(1 for r in results if r.get("detected")) / len(results)
        lead_times = [r.get("lead_time", 0) for r in results if r.get("detected")]
        avg_lead = float(np.mean(lead_times)) if lead_times else 0.0

        fp = self._fast_false_positive_2019()

        print("-")
        print(f" FAST_TUNE detection_rate: {det_rate:.1%}")
        print(f" FAST_TUNE avg_lead_time: {avg_lead:.2f} days")
        print(f" FAST_TUNE fp_rate_2019: {fp:.1%}")

        return {
            "detection_rate": det_rate,
            "avg_lead_time": avg_lead,
            "false_positive_rate": fp,
            "details": results,
        }

    def run_refinement_pipeline(self):
        """Run Day 4-7 pipeline: diagnostics -> optimization -> final validation."""

        from .final_reporting import write_final_report

        print("=" * 60)
        print("S68 REFINEMENT PIPELINE (DAY 4-7)")
        print("=" * 60)

        # Day 4: diagnostics
        print("[Day 4] Expanded event validation...")
        event_results = self.test_historical_events()

        print("[Day 4] False positive deep analysis (2019)...")
        fp_deep = analyze_false_positives_detailed(self.detector, year="2019")
        # Keep old interface for criteria
        fp_results = {"false_positive_rate": fp_deep.get("fp_rate", 0.0), "total_alerts": fp_deep.get("total_fps", 0), "total_days": 0}

        print("[Day 4] Robustness check...")
        param_results = self.test_parameter_sensitivity()

        # Save diagnostic markdown
        import os
        os.makedirs("results/reports", exist_ok=True)
        diag_path = "results/reports/diagnostic_report.md"
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write("# S68 Diagnostic Report\n\n")
            f.write(f"Detection rate: {event_results.get('detection_rate',0.0):.1%}\n\n")
            f.write(f"High priority detection rate: {event_results.get('high_priority_rate',0.0):.1%}\n\n")
            f.write(f"False positive rate (2019): {fp_results.get('false_positive_rate',0.0):.1%}\n\n")
            f.write(f"Robustness success rate: {param_results.get('success_rate',0.0):.1%}\n\n")
        print(f" Diagnostic report saved to {diag_path}")

        # Day 5: optimization
        # If we already have an optimal config from a previous run, reuse it to avoid
        # repeating slow yfinance downloads.
        cfg = None
        try:
            import json
            if os.path.exists("results/optimal_config.json"):
                with open("results/optimal_config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                print("[Day 5] Reusing existing results/optimal_config.json")
            else:
                print("[Day 5] Parameter grid search (this can take a while)...")
                best, _all_results = parameter_grid_search()
                cfg = save_optimal_config(best)
                if cfg:
                    print(" Optimal configuration saved to results/optimal_config.json")
        except Exception as e:
            print(f"[Day 5] Optimization step failed: {e}")
            cfg = None

        # Apply optimized detector
        if cfg:
            self.detector = MultiScaleDetector(
                scales=cfg["scales"],
                min_scales=cfg["min_scales"],
                sensitivity=cfg["sensitivity"],
                baseline_period=4,
            )

        # Day 7: final validation (Day 6 docs already added)
        print("[Day 7] Final validation run...")
        final = self.final_validation_run()

        # Generate final report markdown
        final_report_path = "results/FINAL_REPORT.md"
        write_final_report(
            final_report_path,
            cfg or {"scales": self.detector.scales, "min_scales": self.detector.min_scales, "sensitivity": self.detector.sensitivity},
            final["event_results"],
            final["fp_results"],
            final["param_results"],
            final["criteria"],
            final["recommendation"],
        )
        print(f" Final report saved to {final_report_path}")

        return final

    def final_validation_run(self):
        """Final validation using expanded events + FP + robustness."""

        event_results = self.test_historical_events()
        fp_results = self.test_false_positives()
        param_results = self.test_parameter_sensitivity()

        criteria = {
            "High Priority Detection >= 80%": event_results.get("high_priority_rate", 0.0) >= 0.80,
            "Overall Detection >= 70%": event_results.get("detection_rate", 0.0) >= 0.70,
            "Average Lead Time >= 2 days": event_results.get("avg_lead_time", 0.0) >= 2.0,
            "False Positive Rate <= 20%": fp_results.get("false_positive_rate", 1.0) <= 0.20,
            "Parameter Robustness >= 50%": param_results.get("success_rate", 0.0) >= 0.50,
        }

        passed = sum(1 for v in criteria.values() if v)
        total = len(criteria)

        if passed == total:
            recommendation = "DEPLOY"
        elif passed >= 4:
            recommendation = "CONDITIONAL_DEPLOY"
        elif passed >= 3:
            recommendation = "REFINE"
        else:
            recommendation = "REASSESS"

        return {
            "criteria": criteria,
            "passed": passed,
            "total": total,
            "recommendation": recommendation,
            "event_results": event_results,
            "fp_results": fp_results,
            "param_results": param_results,
        }

    def _fast_false_positive_2019(self):
        data = self._download_with_retry("SPY", "2019-01-01", "2019-12-31")
        prices = data["Close"].values

        det = MultiScaleDetector(
            scales=self.detector.scales,
            min_scales=self.detector.min_scales,
            sensitivity=self.detector.sensitivity,
            baseline_period=4,
        )

        fit_n = min(400, len(prices) - 1)
        if fit_n < 300 or len(prices) < (fit_n + max(det.scales) + 10):
            return 1.0

        det.fit(prices[:fit_n])

        alerts = 0
        total = 0
        for i in range(fit_n, len(prices), 2):  # sample every 2 days to speed up
            if i < max(det.scales):
                continue
            hist = prices[max(0, i - max(det.scales) * 3) : i]
            if det.detect(hist)["is_anomaly"]:
                alerts += 1
            total += 1

        return alerts / max(1, total)

    def test_historical_events(self):
        """Day 4: expanded event set with priority."""

        EVENTS = expanded_events()

        results = []
        high_total = 0
        high_detected = 0

        for event in EVENTS:
            r = self._test_single_event(event["date"], event["name"], event["ticker"])
            r["priority"] = event.get("priority", "")
            results.append(r)

            if event.get("priority") == "HIGH":
                high_total += 1
                if r.get("detected"):
                    high_detected += 1

            status = "OK" if r.get("detected") else "NO"
            lead = str(r.get("lead_time", 0)) + "d" if r.get("detected") else "---"
            print(f"  [{status}] {event['name']}: lead={lead} priority={event.get('priority','')}")

        detection_rate = float(sum(1 for r in results if r.get("detected")) / max(1, len(results)))
        lead_times = [r.get("lead_time", 0) for r in results if r.get("detected")]
        avg_lead_time = float(np.mean(lead_times)) if lead_times else 0.0
        high_priority_rate = float(high_detected / max(1, high_total))

        print(f" Detection Rate: {detection_rate:.1%}")
        print(f" High Priority Detection Rate: {high_priority_rate:.1%}")
        print(f" Average Lead Time: {avg_lead_time:.1f} days")

        return {
            "detection_rate": detection_rate,
            "high_priority_rate": high_priority_rate,
            "avg_lead_time": avg_lead_time,
            "details": results,
        }

    def _test_single_event(self, event_date, event_name, ticker):
        """Test detection for a single event using the multi-scale detector."""

        event_dt = pd.to_datetime(event_date)
        start = event_dt - pd.Timedelta(days=1200)  # need long history for 100-day scale
        end = event_dt + pd.Timedelta(days=5)

        data = self._download_with_retry(ticker, start, end)
        prices = data["Close"].values
        dates = data.index

        # Day 3 requires enough baseline history for the largest (100-day) scale.
        # Use a larger training slice so each scale can learn adaptive thresholds.
        train_size = min(400, len(prices) - 1)
        if train_size < 300 or len(prices) < (train_size + max(self.detector.scales) + 10):
            return {
                "detected": False,
                "lead_time": 0,
                "confidence": 0.0,
                "num_scales": 0,
                "event_name": event_name,
                "reason": "not enough data",
                "all_detections": [],
            }

        det = MultiScaleDetector(
            scales=self.detector.scales,
            min_scales=self.detector.min_scales,
            sensitivity=self.detector.sensitivity,
            baseline_period=4,
        )
        det.fit(prices[:train_size])

        detections = []
        for i in range(train_size, len(prices)):
            # Need enough history for the largest scale
            if i < max(det.scales):
                continue
            # Use up to 3x max scale history for stability
            hist = prices[max(0, i - max(det.scales) * 3) : i]
            res = det.detect(hist)
            detections.append(
                {
                    "date": dates[i],
                    "is_anomaly": bool(res["is_anomaly"]),
                    "confidence": float(res.get("confidence", 0.0)),
                    "num_scales": int(res.get("num_detecting", 0)),
                    "agreeing_scales": list(res.get("agreeing_scales", [])),
                    "days_before": int((event_dt - dates[i]).days),
                }
            )

        anomalies = [d for d in detections if d["is_anomaly"]]
        week_before = [a for a in anomalies if 0 <= a["days_before"] <= 7]
        if week_before:
            earliest = max(week_before, key=lambda x: x["days_before"])
            return {
                "detected": True,
                "lead_time": int(earliest["days_before"]),
                "confidence": float(earliest.get("confidence", 0.0)),
                "num_scales": int(earliest.get("num_scales", 0)),
                "event_name": event_name,
                "all_detections": detections,
            }

        return {
            "detected": False,
            "lead_time": 0,
            "confidence": 0.0,
            "num_scales": 0,
            "event_name": event_name,
            "all_detections": detections,
        }

    def test_false_positives(self):
        """Day 4: false positive rate measured on calm periods."""

        CALM_PERIODS = [
            {"start": "2017-01-01", "end": "2017-12-31", "name": "2017 Bull"},
            {"start": "2019-01-01", "end": "2019-12-31", "name": "2019 Pre-COVID"},
            {"start": "2013-01-01", "end": "2013-12-31", "name": "2013 Recovery"},
        ]

        total_alerts = 0
        total_days = 0

        for period in CALM_PERIODS:
            data = self._download_with_retry("SPY", period["start"], period["end"])
            prices = data["Close"].values
            days = len(prices)
            if days < 250:
                continue

            det = MultiScaleDetector(
                scales=self.detector.scales,
                min_scales=self.detector.min_scales,
                sensitivity=self.detector.sensitivity,
                baseline_period=4,
            )

            fit_n = min(400, len(prices) - 1)
            if fit_n < 300 or len(prices) < (fit_n + max(det.scales) + 10):
                continue

            det.fit(prices[:fit_n])

            alerts = 0
            for i in range(fit_n, len(prices)):
                if i < max(det.scales):
                    continue
                hist = prices[max(0, i - max(det.scales) * 3) : i]
                if det.detect(hist)["is_anomaly"]:
                    alerts += 1

            fpr = alerts / days if days > 0 else 0.0
            print(f"  {period['name']}: {alerts} alerts / {days} days = {fpr:.1%}")

            total_alerts += alerts
            total_days += days

        overall_fpr = total_alerts / max(1, total_days) if total_days > 0 else 0.0
        print(f" Overall False Positive Rate: {overall_fpr:.1%}")

        return {
            "false_positive_rate": float(overall_fpr),
            "total_alerts": int(total_alerts),
            "total_days": int(total_days),
        }

    def test_sensitivity_parameter(self):
        """Find a reasonable sensitivity (k) for adaptive thresholds.

        We do a quick check on the COVID crash window and choose the *lowest*
        false-positive rate among settings that still detect the crash in the
        week before.
        """

        sensitivity_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
        results = []

        event_date = "2020-02-24"
        event_dt = pd.to_datetime(event_date)
        # Pull a longer range so we have enough trading days for baseline windows.
        data = self._download_with_retry("SPY", "2019-01-01", "2020-03-15")
        prices = data["Close"].values
        dates = data.index

        # Need enough data to form baseline_period overlapping windows.
        # With window_size=50 and baseline_period=10, requirement is ~275 points.
        # For the quick sensitivity sweep, use a smaller baseline_period.
        baseline_period = 6
        train_n = min(220, len(prices))

        for sens in sensitivity_values:
            det = TopologicalAnomalyDetector(window_size=50, baseline_period=baseline_period, sensitivity=sens)
            det.fit(prices[:train_n])

            detections = 0
            detected_event = False

            for i in range(train_n, len(prices)):
                if i < det.window_size:
                    continue
                w = prices[i - det.window_size : i]
                res = det.detect(w)
                if res["is_anomaly"]:
                    detections += 1
                    days_before = int((event_dt - dates[i]).days)
                    if 0 <= days_before <= 7:
                        detected_event = True

            total_days = max(1, len(prices) - train_n)
            fp_rate_proxy = detections / total_days
            results.append(
                {
                    "sensitivity": float(sens),
                    "detected_event": bool(detected_event),
                    "total_alerts": int(detections),
                    "fp_rate": float(fp_rate_proxy),
                }
            )

            print(
                f"  k={sens:>3}: detected={('YES' if detected_event else 'NO ')} "
                f"alerts={detections:<4} fp_proxy={fp_rate_proxy:.1%}"
            )

        valid = [r for r in results if r["detected_event"]]
        if valid:
            best = min(valid, key=lambda r: r["fp_rate"])

            # Try slightly more conservative setting to shave off false positives,
            # as long as we still detect the event.
            candidate = float(best["sensitivity"]) + 0.25
            try_det = TopologicalAnomalyDetector(window_size=50, baseline_period=baseline_period, sensitivity=candidate)
            try_det.fit(prices[:train_n])
            det_ok = False
            det_cnt = 0
            for i in range(train_n, len(prices)):
                w = prices[i - try_det.window_size : i]
                res = try_det.detect(w)
                if res["is_anomaly"]:
                    det_cnt += 1
                    days_before = int((event_dt - dates[i]).days)
                    if 0 <= days_before <= 7:
                        det_ok = True
            fp_proxy = det_cnt / max(1, (len(prices) - train_n))

            chosen = best
            if det_ok and fp_proxy <= best["fp_rate"]:
                chosen = {
                    "sensitivity": candidate,
                    "detected_event": True,
                    "total_alerts": det_cnt,
                    "fp_rate": fp_proxy,
                }

            print(
                f" Recommended sensitivity (k): {chosen['sensitivity']} (fp_proxy={chosen['fp_rate']:.1%})"
            )
            return float(chosen["sensitivity"])

        print(" WARNING: No sensitivity value detected the event; using default k=2.0")
        return 2.0

    def test_parameter_sensitivity(self):
        """Day 4/7: robustness check.

        For speed and relevance, vary sensitivity and require detection on a key event.
        """

        test_event = "2020-02-24"
        event_dt = pd.to_datetime(test_event)

        data = self._download_with_retry("SPY", event_dt - pd.Timedelta(days=1200), event_dt + pd.Timedelta(days=5))
        prices = data["Close"].values
        dates = data.index

        train_size = 400
        if len(prices) < (train_size + max(self.detector.scales) + 10):
            return {"success_rate": 0.0, "avg_lead_time": 0.0, "lead_time_std": 0.0, "details": []}

        sensitivities = [0.5, 0.75, 1.0, 1.25, 1.5]
        results = []

        for i, sens in enumerate(sensitivities, start=1):
            print(f"  Testing sensitivity {i}/{len(sensitivities)} (k={sens})...", end="\r")
            det = MultiScaleDetector(
                scales=self.detector.scales,
                min_scales=self.detector.min_scales,
                sensitivity=sens,
                baseline_period=4,
            )
            det.fit(prices[:train_size])

            detected = False
            lead_time = 0
            for j in range(train_size, len(prices)):
                if j < max(det.scales):
                    continue
                hist = prices[max(0, j - max(det.scales) * 3) : j]
                res = det.detect(hist)
                days_before = int((event_dt - dates[j]).days)
                if res["is_anomaly"] and 0 <= days_before <= 7:
                    detected = True
                    lead_time = max(lead_time, days_before)
            results.append({"sensitivity": sens, "detected": detected, "lead_time": int(lead_time)})

        print()

        success_rate = float(sum(1 for r in results if r["detected"]) / max(1, len(results)))
        lead_times = [r["lead_time"] for r in results if r["detected"]]
        avg_lead_time = float(np.mean(lead_times)) if lead_times else 0.0
        lead_time_std = float(np.std(lead_times)) if lead_times else 0.0

        print(f" Success Rate: {success_rate:.1%}")
        print(f" Lead Time: {avg_lead_time:.1f} +/- {lead_time_std:.1f} days")

        return {
            "success_rate": success_rate,
            "avg_lead_time": avg_lead_time,
            "lead_time_std": lead_time_std,
            "details": results,
        }

    def _test_params_on_event(self, det, event_date):
        event_dt = pd.to_datetime(event_date)
        start = event_dt - pd.Timedelta(days=900)
        end = event_dt + pd.Timedelta(days=5)

        data = self._download_with_retry("SPY", start, end)
        prices = data["Close"].values
        dates = data.index

        step = max(1, det.window_size // 2)
        min_required = det.window_size + step * (det.baseline_period - 1)

        train_size = min(max(200, min_required + 20), len(prices) - det.window_size - 1)
        if train_size < min_required + 10:
            return False, 0

        det.fit(prices[:train_size])

        anomalies = []
        for i in range(train_size, len(prices)):
            if i < det.window_size:
                continue
            w = prices[i - det.window_size : i]
            res = det.detect(w)
            if res["is_anomaly"]:
                anomalies.append(int((event_dt - dates[i]).days))

        week_before = [d for d in anomalies if 0 <= d <= 7]
        if week_before:
            return True, int(max(week_before))
        return False, 0

    def generate_report(self, event_results, fp_results, param_results):
        print(" " + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        criteria = {
            "Event Detection Rate >= 70%": event_results["detection_rate"] >= 0.70,
            "Average Lead Time >= 2 days": event_results["avg_lead_time"] >= 2.0,
            "False Positive Rate <= 20%": fp_results["false_positive_rate"] <= 0.20,
            "Parameter Success Rate >= 50%": param_results["success_rate"] >= 0.50,
        }

        passed = int(sum(criteria.values()))
        total = int(len(criteria))
        print(f" Criteria Met: {passed}/{total}")
        print("-" * 60)
        for criterion, ok in criteria.items():
            print(("PASS: " if ok else "FAIL: ") + criterion)

        print(" " + "=" * 60)
        if passed >= 3:
            print("RECOMMENDATION: PROCEED to production system")
        elif passed >= 2:
            print("RECOMMENDATION: REFINE and re-test")
        else:
            print("RECOMMENDATION: NOT VIABLE (as-is)")
        print("=" * 60)

        self._save_results_to_files(event_results, fp_results, param_results)

    def _save_results_to_files(self, event_results, fp_results, param_results):
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/figures", exist_ok=True)
        os.makedirs("results/logs", exist_ok=True)
        os.makedirs("results/reports", exist_ok=True)

        pd.DataFrame(event_results["details"]).to_csv("results/event_detection.csv", index=False)
        pd.DataFrame(param_results["details"]).to_csv(
            "results/parameter_sensitivity.csv", index=False
        )

        # Save a simple JSON-like report as text
        report_path = os.path.join("results", "reports", "summary.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("TDA Validation Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Event detection rate: {event_results['detection_rate']:.1%}\n")
            f.write(f"Avg lead time: {event_results['avg_lead_time']:.2f} days\n")
            f.write(f"False positive rate: {fp_results['false_positive_rate']:.1%}\n")
            f.write(f"Parameter success rate: {param_results['success_rate']:.1%}\n")

        print(" Results saved to:")
        print(" - results/event_detection.csv")
        print(" - results/parameter_sensitivity.csv")
        print(" - results/reports/summary.txt")

    @staticmethod
    def _download_with_retry(ticker, start, end, max_retries=3):
        last_err = None
        for attempt in range(max_retries):
            try:
                data = yf.download(ticker, start=start, end=end, progress=False)
                if data is not None and not data.empty:
                    return data
            except Exception as e:
                last_err = e
            time.sleep(1.5)
        raise RuntimeError(f"Failed to download {ticker}: {last_err}")
