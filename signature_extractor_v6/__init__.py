"""
Signature Extractor v3 - LLM-powered signature extraction from PDF documents.

A modular framework for extracting signature metadata from modern slavery statements
and other PDF documents using Large Language Models including open-source alternatives
and local models via Ollama.

Key Features:
- LLM-based signature detection and analysis
- Multiple input sources (CSV, directory, URLs)
- Support for commercial, open-source vision-language models, and local Ollama models
- Structured output matching signature_columns.xlsx format
- Progress tracking and comprehensive logging
- Both CLI and programmatic interfaces

Example Usage:
    # Simple programmatic usage with OpenAI
    from signature_extractor_v3 import extract_signatures_from_csv
    results = extract_signatures_from_csv("statements.csv", "output/")
    
    # Using GLM-4.5V model
    results = extract_signatures_from_csv(
        "statements.csv", "output/", 
        llm_provider="huggingface", 
        llm_model="zai-org/GLM-4.5V:novita"
    )
    
    # Using Ollama Moondream model
    results = extract_signatures_from_csv(
        "statements.csv", "output/", 
        llm_provider="ollama", 
        llm_model="moondream:1.8b"
    )
    
    # Using preset configurations
    from signature_extractor_v3 import ExtractionConfig
    config = ExtractionConfig.create_preset("ollama-moondream", output_dir="results/")
    
    # Advanced usage with custom configuration
    from signature_extractor_v3 import (
        ExtractionConfig, LLMConfig, ProcessingConfig,
        ExtractionOrchestrator, CSVSourceAdapter
    )
    
    config = ExtractionConfig(
        llm=LLMConfig(
            provider="ollama",
            model="moondream:1.8b",
            ollama_base_url="http://localhost:11434"
        ),
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
    hf_token: str = None,
    base_url: str = None,
    ollama_url: str = "http://localhost:11434",
    **kwargs
) -> list:
    """
    Simple interface for extracting signatures from CSV file.
    
    Args:
        csv_path: Path to CSV file with PDF URLs
        output_dir: Output directory for results
        llm_provider: LLM provider ("openai", "anthropic", "huggingface", "ollama")
        llm_model: LLM model to use
        api_key: API key for LLM provider
        hf_token: HuggingFace token (for huggingface provider)
        base_url: Base URL for API (for huggingface provider)
        ollama_url: Ollama server URL (for ollama provider)
        **kwargs: Additional configuration options
        
    Returns:
        List of extraction results
    """
    config = ExtractionConfig(
        llm=LLMConfig(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key,
            hf_token=hf_token,
            base_url=base_url,
            ollama_base_url=ollama_url
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
    hf_token: str = None,
    base_url: str = None,
    ollama_url: str = "http://localhost:11434",
    **kwargs
) -> list:
    """
    Simple interface for extracting signatures from directory of PDFs.
    
    Args:
        directory_path: Path to directory containing PDFs
        output_dir: Output directory for results
        llm_provider: LLM provider ("openai", "anthropic", "huggingface", "ollama")
        llm_model: LLM model to use
        api_key: API key for LLM provider
        hf_token: HuggingFace token (for huggingface provider)
        base_url: Base URL for API (for huggingface provider)
        ollama_url: Ollama server URL (for ollama provider)
        **kwargs: Additional configuration options
        
    Returns:
        List of extraction results
    """
    from .adapters.directory_adapter import DirectorySourceAdapter
    
    config = ExtractionConfig(
        llm=LLMConfig(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key,
            hf_token=hf_token,
            base_url=base_url,
            ollama_base_url=ollama_url
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


__version__ = "3.0.0"
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
    
    # Utilities
    "ProgressTracker",
    "setup_logging",
    "load_prompt",
    
    # Convenience functions
    "extract_signatures_from_csv",
    "extract_signatures_from_directory"
]