import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

summary_path = RES / "reports" / "summary.txt"
opt_path = RES / "optimal_config.json"
ev_path = RES / "event_detection.csv"
ps_path = RES / "parameter_sensitivity.csv"

summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
opt = json.loads(opt_path.read_text(encoding="utf-8")) if opt_path.exists() else {}

# Parse the simple summary metrics (fallbacks)
metrics = {
    "event_detection_rate": None,
    "avg_lead_time": None,
    "false_positive_rate": None,
    "parameter_success_rate": None,
}
for line in summary_text.splitlines():
    line = line.strip()
    if line.lower().startswith("event detection rate:"):
        metrics["event_detection_rate"] = float(line.split(":", 1)[1].strip().strip("%")) / 100.0
    if line.lower().startswith("avg lead time:"):
        metrics["avg_lead_time"] = float(line.split(":", 1)[1].strip().split()[0])
    if line.lower().startswith("false positive rate:"):
        metrics["false_positive_rate"] = float(line.split(":", 1)[1].strip().strip("%")) / 100.0
    if line.lower().startswith("parameter success rate:"):
        metrics["parameter_success_rate"] = float(line.split(":", 1)[1].strip().strip("%")) / 100.0

# Parse event details (CSV has a huge JSON column; we only need first columns)
# Use python engine and only read needed columns to avoid memory issues.
ev_df = pd.read_csv(ev_path, usecols=["detected", "lead_time", "confidence", "num_scales", "event_name"], engine="python")

# Parse parameter sensitivity
ps_df = pd.read_csv(ps_path)

# Compute metrics from data where possible
if metrics["event_detection_rate"] is None and len(ev_df) > 0:
    metrics["event_detection_rate"] = float(ev_df["detected"].mean())
if metrics["avg_lead_time"] is None and len(ev_df) > 0:
    lt = ev_df.loc[ev_df["detected"], "lead_time"]
    metrics["avg_lead_time"] = float(lt.mean()) if len(lt) else 0.0

# Robustness (success rate)
if metrics["parameter_success_rate"] is None and len(ps_df) > 0:
    metrics["parameter_success_rate"] = float(ps_df["detected"].mean())

# Criteria (using the original thresholds; high-priority not available from these stored results)
def _val(key, default):
    v = metrics.get(key)
    return default if v is None else v

criteria = {
    "Overall Detection >= 70%": _val("event_detection_rate", 0.0) >= 0.70,
    "Average Lead Time >= 2 days": _val("avg_lead_time", 0.0) >= 2.0,
    "False Positive Rate <= 20%": _val("false_positive_rate", 1.0) <= 0.20,
    "Parameter Robustness >= 50%": _val("parameter_success_rate", 0.0) >= 0.50,
}

passed = sum(1 for v in criteria.values() if v)

# Mirror earlier behavior: proceed if core live metrics look good even if robustness fails
if criteria["Overall Detection >= 70%"] and criteria["False Positive Rate <= 20%"] and criteria["Average Lead Time >= 2 days"]:
    if criteria["Parameter Robustness >= 50%"]:
        recommendation = "PROCEED to production system"
    else:
        recommendation = "PROCEED to production system (robustness FAIL – monitor/tune)"
else:
    recommendation = "REFINE and re-test"

lines = []
lines.append("# S68 FINAL VALIDATION REPORT")
lines.append("")
lines.append(f"Recommendation: {recommendation}")
lines.append("")
lines.append("## Optimal Configuration Used")
lines.append("```json")
lines.append(json.dumps(opt, indent=2))
lines.append("```")
lines.append("")
lines.append("## Summary Metrics")
lines.append(f"- Event detection rate: {_val('event_detection_rate', 0.0):.1%}")
lines.append(f"- Avg lead time: {_val('avg_lead_time', 0.0):.2f} days")
lines.append(f"- False positive rate: {_val('false_positive_rate', 0.0):.1%}")
lines.append(f"- Parameter success rate: {_val('parameter_success_rate', 0.0):.1%}")
lines.append("")
lines.append("## Criteria")
for k, ok in criteria.items():
    lines.append(f"- [{'PASS' if ok else 'FAIL'}] {k}")
lines.append("- [N/A] High Priority Detection >= 80% (not recorded in stored CSV outputs)")
lines.append("")
lines.append("## Event Results")
for _, r in ev_df.iterrows():
    status = "DETECTED" if bool(r["detected"]) else "MISSED"
    lead = int(r["lead_time"]) if bool(r["detected"]) else "-"
    lines.append(f"- {status}: {r['event_name']} (lead={lead}d, conf={float(r['confidence']):.2f}, scales={int(r['num_scales'])})")

out_path = RES / "FINAL_REPORT.md"
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {out_path}")
