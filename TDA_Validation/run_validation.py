"""Main entry point for the validation suite.

Usage:
  python run_validation.py

This will:
  1) Download required market data (via yfinance)
  2) Run validation tests
  3) Generate report + save outputs under results/

Expected runtime: ~10-15 minutes (depends on network + CPU).
"""

from src.validator import ValidationSuite


if __name__ == "__main__":
    print("Initializing validation suite...")
    suite = ValidationSuite()

    # Full refinement pipeline (Day 4-7). This may take a while.
    suite.run_refinement_pipeline()
