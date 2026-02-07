# USAGE

This document describes how to run the validation suite and generate reports.

## Quickstart (Windows)

From PowerShell or cmd.exe:

1) Create a virtual environment and install dependencies

```bat
cd C:\S68\TDA_Validation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Run unit tests

```bat
pytest -q
```

3) Run the full validation

```bat
python run_validation.py
```

Outputs are written under `results/`.

## Package Entry Point

A thin production namespace package is provided under `s68/`.

Run via module:

```bat
python -m s68.tda_validation
```

This delegates to the existing `src.validator.ValidationSuite`.

## Generate FINAL_REPORT.md

A small generator script is provided:

```bat
python scripts\generate_final_report.py
```

This reads the latest artifacts under `results/` (if present) and writes
`FINAL_REPORT.md` at the repository root.

If `results/` is missing, the generator will still produce a scaffolded report
with placeholders.

## Common Issues

- If downloads fail, re-run later or confirm your network access.
- If you want to run without downloading, ensure cached data exists under
  `data/`.

## Files

- `docs/THEORY.md` - Background and pipeline description
- `docs/USAGE.md`  - How to run and generate reports
- `HANDOFF.md`     - Notes for the next person
- `COMPLETION_CHECKLIST.md` - What is done vs remaining
