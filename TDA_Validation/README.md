# TDA Validation (S68)

Proof-of-Concept Validation for Topological Data Analysis (TDA) in Financial Markets.

## Goal
Validate the hypothesis that **topological structure changes 1–7 days before major market events** and can be detected via **persistent homology**.

## Quickstart (Windows)

### 1) Create venv + install deps
```bat
cd C:\S68\TDA_Validation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run unit tests
```bat
pytest -q
```

### 3) Run full validation
```bat
python run_validation.py
```

Outputs:
- `results/event_detection.csv`
- `results/parameter_sensitivity.csv`
- Figures in `results/figures/`

## Project Structure
- `src/embedding.py` — Takens embedding + AMI delay selection
- `src/topology.py` — persistent homology feature extraction
- `src/detector.py` — anomaly detection vs baseline
- `src/validator.py` — full validation suite + report + saved artifacts

## Notes
- Market data downloaded via `yfinance` and cached under `data/`.
- Default settings are chosen for daily SPY data; adjust window sizes and thresholds if needed.
