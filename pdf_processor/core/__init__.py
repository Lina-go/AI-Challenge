# pdf_processor_v3/core/__init__.py - CORREGIR imports:

"""
Core processing modules for PDF pipeline.
"""

from .llm_interface import LLMInterface, create_llm_interface
from .page_processor import PageProcessor
from .document_processor import DocumentProcessor
from .signature_analyzer import SignatureAnalyzer
from .orchestrator import MainOrchestrator, SignatureOrchestrator, DocumentOrchestrator

__all__ = [
    "LLMInterface", 
    "create_llm_interface",
    "PageProcessor", 
    "DocumentProcessor",
    "SignatureAnalyzer",
    "MainOrchestrator",
    "SignatureOrchestrator", 
    "DocumentOrchestrator"
]