"""
Core processing components for signature extraction.

This package contains the main processing engine with focused components:

- ExtractionOrchestrator: Coordinates the complete extraction workflow
- DocumentProcessor: Handles PDF to image conversion and document processing
- SignatureAnalyzer: LLM-based signature detection and analysis
- DataProcessor: Data transformation, validation, and cleaning
- ResultManager: Results collection, formatting, and persistence

Each component follows the Single Responsibility Principle and can be
used independently or as part of the orchestrated workflow.
"""

from .orchestrator import ExtractionOrchestrator
from .document_processor import DocumentProcessor
from .signature_analyzer import SignatureAnalyzer
from .data_processor import DataProcessor
from .result_manager import ResultManager

__all__ = [
    "ExtractionOrchestrator",
    "DocumentProcessor", 
    "SignatureAnalyzer",
    "DataProcessor",
    "ResultManager"
]

# Package-level constants
DEFAULT_DPI = 300
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_WORKERS = 4

# Component version tracking for debugging
COMPONENT_VERSIONS = {
    "orchestrator": "2.0.0",
    "document_processor": "2.0.0", 
    "signature_analyzer": "2.0.0",
    "data_processor": "2.0.0",
    "result_manager": "2.0.0"
}

def get_component_info():
    """Get information about core components."""
    return {
        "components": list(__all__),
        "versions": COMPONENT_VERSIONS,
        "default_settings": {
            "dpi": DEFAULT_DPI,
            "timeout": DEFAULT_TIMEOUT,
            "max_workers": DEFAULT_MAX_WORKERS
        }
    }