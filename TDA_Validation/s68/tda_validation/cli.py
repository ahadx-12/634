"""CLI helpers for TDA Validation.

This is intentionally small: it delegates to the existing implementation.
"""

from __future__ import annotations

import argparse

from .api import ValidationSuite


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="s68-tda-validation",
        description="Run the S68 TDA validation suite.",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download market data (use local cache).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    suite = ValidationSuite()

    # Additive behavior: only call optional method if it exists.
    if args.no_download and hasattr(suite, "disable_download"):
        suite.disable_download()

    suite.run_full_validation()
    return 0
