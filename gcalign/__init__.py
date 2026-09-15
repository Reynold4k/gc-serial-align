"""Align serial immunofluorescence sections into a spatial-transcriptomics frame.

The pipeline runs in seven stages; see `gcalign.cli` or the project README.
"""
from .config import Config, load

__all__ = ["Config", "load", "__version__"]
__version__ = "0.1.0"
