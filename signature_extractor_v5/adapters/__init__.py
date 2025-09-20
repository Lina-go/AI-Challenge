# signature_extractor_v5/adapters/__init__.py
"""Source adapters for different input types."""

from .csv_adapter import CSVSourceAdapter
from .directory_adapter import DirectorySourceAdapter

__all__ = ["CSVSourceAdapter", "DirectorySourceAdapter"]