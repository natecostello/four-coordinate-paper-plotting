"""
fcp-plotting: Matplotlib utilities for Four-Coordinate Paper pseudo-velocity plots.

This package provides helper functions to create properly formatted matplotlib plots
for shock response spectrum analysis using Four-Coordinate Paper (FCP) conventions.
"""

from .fcp import fcp

try:
    from ._version import version as __version__
except ImportError:
    # Fallback version for development installs without setuptools_scm
    __version__ = "dev"

__all__ = [
    "fcp",
    "__version__",
]

