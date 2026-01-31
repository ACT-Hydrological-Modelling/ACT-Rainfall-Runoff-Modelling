"""Top-level package for SCE-UA."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from sceua.sceua import Result, minimize

try:
    __version__ = version("sceua")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "Result",
    "__version__",
    "minimize",
]
