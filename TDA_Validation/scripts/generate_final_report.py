"""Generate FINAL_REPORT.md from the latest results.

This script is intentionally lightweight and robust against partially-written
CSV artifacts.

Usage:
  python scripts/generate_final_report.py

It will write:
  FINAL_REPORT.md

No core math is executed here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
REPORT_PATH = REPO_ROOT / "FINAL_REPORT.md"


@dataclass
class SensitivityRow:
    sensitivity: str
    detected: str
    lead_time: str


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def load_summary() -> str:
    return _read_text(RESULTS_DIR / "reports" / "summary.txt").strip()


def load_parameter_sensitivity(max_rows: int = 20) -> list[SensitivityRow]:
    rows: list[SensitivityRow] = []
    path = RESULTS_DIR / "parameter_sensitivity.csv"
    if not path.exists():
        return rows

    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(
                    SensitivityRow(
                        sensitivity=str(r.get("sensitivity", "")),
                        detected=str(r.get("detected", "")),
                        lead_time=str(r.get("lead_time", "")),
                    )
                )
    except Exception:
        # Keep robust: if the file is malformed we still generate the report.
        return []

    return rows


def format_sensitivity_table(rows: list[SensitivityRow]) -> str:
    if not rows:
        return "(No readable parameter_sensitivity.csv rows found.)"

    lines = [
        "| sensitivity | detected | lead_time |",
        "|---:|:---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r.sensitivity} | {r.detected} | {r.lead_time} |")
    return "\n".join(lines)


def artifact_status() -> list[str]:
    paths = [
        RESULTS_DIR / "event_detection.csv",
        RESULTS_DIR / "parameter_sensitivity.csv",
        RESULTS_DIR / "reports" / "summary.txt",
    ]

    out: list[str] = []
    for p in paths:
        if p.exists():
            out.append(f"- OK: {p.relative_to(REPO_ROOT)} ({p.stat().st_size} bytes)")
        else:
            out.append(f"- MISSING: {p.relative_to(REPO_ROOT)}")
    return out


def generate() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = load_summary()
    sensitivity_rows = load_parameter_sensitivity()

    # Note: event_detection.csv in this repo may contain large serialized objects.
    # We do not attempt to parse it here.

    lines: list[str] = []
    lines.append("# FINAL REPORT - TDA Validation (S68)")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    if summary:
        lines.append("Raw summary (from results/reports/summary.txt):")
        lines.append("")
        lines.append("```text")
        lines.append(summary)
        lines.append("```")
    else:
        lines.append("Summary not found. Run `python run_validation.py` to generate results.")

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.extend(artifact_status())

    lines.append("")
    lines.append("## Parameter Sensitivity (Preview)")
    lines.append("")
    lines.append(format_sensitivity_table(sensitivity_rows))

    lines.append("")
    lines.append("## Method (High Level)")
    lines.append("")
    lines.append("- Takens delay embedding of a 1D price series")
    lines.append("- Persistent homology feature extraction")
    lines.append("- Anomaly detection vs a baseline window")
    lines.append("- Validation against known historical events")

    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Results depend on event definitions, parameter choices, and data quality.")
    lines.append("- This repo is a validation harness and is not investment advice.")

    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    lines.append("- Improve event_detection.csv format to a strict CSV with stable columns.")
    lines.append("- Add plots and embed key figures in this report.")
    lines.append("- Expand the parameter sweep and add cross-validation.")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    REPORT_PATH.write_text(generate(), encoding="utf-8", errors="strict")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
