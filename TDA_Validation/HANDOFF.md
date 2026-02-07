# HANDOFF

This repo contains a proof-of-concept validation harness for using persistent
homology features as early warning signals for market events.

## What Works Today

- Validation suite runs via:
  - `python run_validation.py`
  - `python -m s68.tda_validation` (thin wrapper over `src/`)

- Results are written under `results/`.

- A report scaffold can be generated via:
  - `python scripts/generate_final_report.py`

## Where Things Live

- Core implementation: `src/`
  - embedding, topology, detector, validator

- Production import namespace (wrapper): `s68/`
  - delegates to `src/` without changing math

- Documentation: `docs/`

## Known Issues / Notes

- `results/event_detection.csv` may not be a strict CSV. It may contain
  serialized Python objects. The final report generator does not parse it.

- If download is slow or blocked, ensure cached data exists under `data/`.

## Next Person - Suggested Tasks

- Make the event detection artifact a strict CSV or JSON for easier parsing.
- Add a formal package build (pyproject) if distribution is needed.
- Add figures and embed them into FINAL_REPORT.md.
