"""
PDF Processing Pipeline for Modern Slavery Statements.

A modular pipeline for extracting structured content from PDF documents
using LLM-based analysis and content extraction.
"""

from .core.document_processor import DocumentProcessor
from .core.llm_interface import LLMInterface
from .core.gemma_interface import GemmaLLMInterface, create_gemma_interface
from .models.document_result import DocumentResult

__version__ = "0.1.0"
__all__ = ["DocumentProcessor", "LLMInterface", "GemmaLLMInterface", "create_gemma_interface", "DocumentResult"]
