# signature_extractor_v5/__init__.py
"""
Signature Extractor v5 - Docling VLM-powered signature extraction.

Uses IBM Granite Docling Vision-Language Model for signature detection
and extraction from PDF documents.
"""

__version__ = "5.0.0"
__author__ = "Signature Extractor Team"

from .config import ExtractionConfig, DoclingConfig, ProcessingConfig
from .core import ExtractionOrchestrator
from .adapters import CSVSourceAdapter, DirectorySourceAdapter

__all__ = [
    "ExtractionConfig",
    "DoclingConfig", 
    "ProcessingConfig",
    "ExtractionOrchestrator",
    "CSVSourceAdapter",
    "DirectorySourceAdapter"
]