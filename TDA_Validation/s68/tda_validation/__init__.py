"""TDA Validation (S68).

This package is a thin compatibility layer over the existing implementation
in `src/`.

Preferred entry point:
  python -m s68.tda_validation

Or programmatic:
  from s68.tda_validation import ValidationSuite
"""

from .api import ValidationSuite

__all__ = ["ValidationSuite"]
