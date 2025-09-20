"""
Signature Extractor v2 - LLM-powered signature extraction from PDF documents.

A modular framework for extracting signature metadata from modern slavery statements
and other PDF documents using Large Language Models.

Key Features:
- LLM-based signature detection and analysis
- Multiple input sources (CSV, directory, URLs)
- Structured output matching signature_columns.xlsx format
- Progress tracking and comprehensive logging
- Both CLI and programmatic interfaces

Example Usage:
    # Simple programmatic usage
    from signature_extractor_v2 import extract_signatures_from_csv
    results = extract_signatures_from_csv("statements.csv", "output/")
    
    # Advanced usage with custom configuration
    from signature_extractor_v2 import (
        ExtractionConfig, LLMConfig, ProcessingConfig,
        ExtractionOrchestrator, CSVSourceAdapter
    )
    
    config = ExtractionConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o"),
        processing=ProcessingConfig(output_dir="results/")
    )
    
    adapter = CSVSourceAdapter(config, "data.csv")
    orchestrator = ExtractionOrchestrator(config)
    
    sources = adapter.get_pdf_sources()
    for source in sources:
        source['pdf_path'] = adapter.prepare_source(source)
    
    results = orchestrator.process_multiple_sources(sources)
"""

from .config import (
    ExtractionConfig,
    LLMConfig,
    ProcessingConfig
)

from .base_extractor import (
    BaseSourceAdapter,
    BaseExtractor
)

from .core import (
    ExtractionOrchestrator,
    DocumentProcessor,
    SignatureAnalyzer,
    DataProcessor,
    ResultManager
)

from .adapters import (
    CSVSourceAdapter,
    DirectorySourceAdapter,
    URLSourceAdapter
)

from .models import (
    ExtractionResult,
    SignatureData,
    DocumentMetadata
)

from .utils import (
    ProgressTracker,
    setup_logging,
    load_prompt
)

# Convenience functions for common use cases
from .core.orchestrator import ExtractionOrchestrator
from .adapters.csv_adapter import CSVSourceAdapter

def extract_signatures_from_csv(
    csv_path: str, 
    output_dir: str = "signature_results",
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
    api_key: str = None,
    **kwargs
) -> list:
    """
    Simple interface for extracting signatures from CSV file.
    
    Args:
        csv_path: Path to CSV file with PDF URLs
        output_dir: Output directory for results
        llm_provider: LLM provider ("openai", "anthropic")
        llm_model: LLM model to use
        api_key: API key for LLM provider
        **kwargs: Additional configuration options
        
    Returns:
        List of extraction results
    """
    config = ExtractionConfig(
        llm=LLMConfig(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key
        ),
        processing=ProcessingConfig(
            output_dir=output_dir,
            **kwargs.get('processing_config', {})
        )
    )
    
    adapter = CSVSourceAdapter(config, csv_path)
    orchestrator = ExtractionOrchestrator(config)
    
    sources = adapter.get_pdf_sources()
    for source in sources:
        source['pdf_path'] = adapter.prepare_source(source)
    
    return orchestrator.process_multiple_sources(sources)


def extract_signatures_from_directory(
    directory_path: str,
    output_dir: str = "signature_results",
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
    api_key: str = None,
    **kwargs
) -> list:
    """
    Simple interface for extracting signatures from directory of PDFs.
    
    Args:
        directory_path: Path to directory containing PDFs
        output_dir: Output directory for results
        llm_provider: LLM provider ("openai", "anthropic")
        llm_model: LLM model to use
        api_key: API key for LLM provider
        **kwargs: Additional configuration options
        
    Returns:
        List of extraction results
    """
    from .adapters.directory_adapter import DirectorySourceAdapter
    
    config = ExtractionConfig(
        llm=LLMConfig(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key
        ),
        processing=ProcessingConfig(
            output_dir=output_dir,
            **kwargs.get('processing_config', {})
        )
    )
    
    adapter = DirectorySourceAdapter(config, directory_path)
    orchestrator = ExtractionOrchestrator(config)
    
    sources = adapter.get_pdf_sources()
    for source in sources:
        source['pdf_path'] = adapter.prepare_source(source)
    
    return orchestrator.process_multiple_sources(sources)


__version__ = "2.0.0"
__author__ = "Signature Extractor Team"
__email__ = "support@example.com"

__all__ = [
    # Configuration classes
    "ExtractionConfig",
    "LLMConfig", 
    "ProcessingConfig",
    
    # Base classes
    "BaseSourceAdapter",
    "BaseExtractor",
    
    # Core components
    "ExtractionOrchestrator",
    "DocumentProcessor",
    "SignatureAnalyzer", 
    "DataProcessor",
    "ResultManager",
    
    # Source adapters
    "CSVSourceAdapter",
    "DirectorySourceAdapter",
    "URLSourceAdapter",
    
    # Data models
    "ExtractionResult",
    "SignatureData",
    "DocumentMetadata",
    
    # Utilities
    "ProgressTracker",
    "setup_logging",
    "load_prompt",
    
    # Convenience functions
    "extract_signatures_from_csv",
    "extract_signatures_from_directory"
]