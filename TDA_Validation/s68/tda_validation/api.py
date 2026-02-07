"""Public API wrappers.

The implementation is in `src/`. This module exists to provide a stable
package name for production use.
"""

from __future__ import annotations

# Re-export legacy implementation.
from src.validator import ValidationSuite  # noqa: F401
