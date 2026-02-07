"""Module entry point.

Allows:
  python -m s68.tda_validation

This calls the existing ValidationSuite from `src/`.
"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
