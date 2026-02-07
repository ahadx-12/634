# THEORY

This document describes the theory behind this validation suite.

## Hypothesis

Market time series can exhibit measurable changes in geometric and topological
structure in the days leading up to large market events. The working hypothesis
for this project is:

- Topological structure changes 1 to 7 days before major market events.
- Persistent homology features computed on a delay embedded time series can
  be used as an early warning signal.

This repo is a validation harness for that hypothesis.

## Pipeline Overview

1. Data
   - Daily market data is downloaded via `yfinance` and cached under `data/`.

2. Delay Embedding (Takens)
   - Convert a 1D time series into a point cloud in R^m using delay embedding.
   - Key parameters:
     - delay (tau)
     - embedding dimension (m)

3. Persistent Homology
   - Build a filtration over the embedded point cloud.
   - Compute persistence diagrams and summarize them with features.
   - Typical features include counts and statistics of lifetimes.

4. Detection
   - Compare current topological features to a baseline distribution.
   - Produce anomaly scores and detection decisions.

5. Validation
   - Evaluate detection rate, lead time, and false positive rate.
   - Perform a basic parameter sensitivity sweep.

## Notes on Interpretation

- Persistent homology is sensitive to scaling and noise; preprocessing choices
  matter.
- Lead time results depend on how events are defined and how detection windows
  are chosen.
- This project is focused on reproducible validation, not on making trading
  recommendations.

## Where the Math Lives

The core implementation currently lives in `src/`:

- `src/embedding.py`
- `src/topology.py`
- `src/detector.py`
- `src/validator.py`

The production namespace package `s68/` is a thin wrapper that re-exports the
existing implementation.
