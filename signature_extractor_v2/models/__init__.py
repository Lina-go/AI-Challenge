"""
Data models and structures for signature extraction.
"""

from .extraction_result import ExtractionResult, SourceResult, ProcessingSummary
from .signature_data import SignatureData, SignatureMetadata, SignatoryInfo, DateInfo
from .document_metadata import DocumentMetadata, PageInfo, ProcessingStats

__all__ = [
    "ExtractionResult",
    "SourceResult", 
    "ProcessingSummary",
    "SignatureData",
    "SignatureMetadata",
    "SignatoryInfo",
    "DateInfo",
    "DocumentMetadata",
    "PageInfo",
    "ProcessingStats"
]