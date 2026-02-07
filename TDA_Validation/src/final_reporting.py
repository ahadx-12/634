"""Final reporting helpers for Day 7."""

from __future__ import annotations

import os
from datetime import datetime


def write_final_report(path, config, event_results, fp_results, param_results, criteria, recommendation):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    lines = []
    lines.append("# S68 FINAL VALIDATION REPORT")
    lines.append("")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Recommendation: {recommendation}")
    lines.append("")

    lines.append("## Configuration")
    lines.append("```")
    lines.append(str(config))
    lines.append("```")
    lines.append("")

    lines.append("## Summary Metrics")
    lines.append(f"Overall detection rate: {event_results.get('detection_rate', 0.0):.1%}")
    lines.append(f"High priority detection rate: {event_results.get('high_priority_rate', 0.0):.1%}")
    lines.append(f"Avg lead time: {event_results.get('avg_lead_time', 0.0):.2f} days")
    lines.append(f"False positive rate: {fp_results.get('false_positive_rate', 0.0):.1%}")
    lines.append(f"Robustness success rate: {param_results.get('success_rate', 0.0):.1%}")
    lines.append("")

    lines.append("## Criteria")
    for k, ok in criteria.items():
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {k}")
    lines.append("")

    lines.append("## Event Detail")
    for ev in event_results.get('details', []):
        status = 'DETECTED' if ev.get('detected') else 'MISSED'
        lead = ev.get('lead_time', 0)
        pr = ev.get('priority', '')
        lines.append(f"- {status}: {ev.get('event_name','')} lead={lead} priority={pr}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
