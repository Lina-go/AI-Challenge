"""
PDF Processor - Unified document analysis and signature extraction framework
"""

from .config import (
    ProcessorConfig,
    LLMConfig,
    ProcessingConfig,
    TaskConfig
)

from .core import (
    DocumentProcessor,
    SignatureAnalyzer,
    LLMInterface
)

from .core.llm_factory import (
    create_llm_interface,
    get_available_providers,
    get_recommended_provider
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
def process_document(
    pdf_path: str,
    output_path: str = None,
    llm_provider: str = "openai",
    llm_model: str = None,
    **kwargs
):
    """
    Simple interface for processing a single PDF document.
    
    Args:
        pdf_path: Path to PDF file
        output_path: Output file path (optional)
        llm_provider: LLM provider to use
        llm_model: LLM model to use
        **kwargs: Additional configuration options
        
    Returns:
        DocumentResult object
    """
    from .core.document_processor import DocumentProcessor
    from .core.llm_factory import create_llm_interface
    
    # Create configuration
    config = ProcessorConfig.create_preset(
        llm_provider if llm_provider in ["openai", "anthropic"] else "openai"
    )
    
    if llm_model:
        config.llm.model = llm_model
    if llm_provider != "openai":
        config.llm.provider = llm_provider
    
    # Process document
    llm = create_llm_interface(config.llm)
    processor = DocumentProcessor(llm)
    result = processor.process_document(pdf_path)
    
    # Save if output path specified
    if output_path:
        processor.save_results(result, output_path)
    
    return result

def extract_signatures_from_csv(
    csv_path: str,
    output_dir: str = "signature_results",
    llm_provider: str = "openai",
    llm_model: str = None,
    **kwargs
):
    """
    Simple interface for extracting signatures from CSV file.
    
    Args:
        csv_path: Path to CSV file with PDF URLs
        output_dir: Output directory for results
        llm_provider: LLM provider to use
        llm_model: LLM model to use
        **kwargs: Additional configuration options
        
    Returns:
        List of extraction results
    """
    from .main import SignatureOrchestrator
    
    # Create configuration
    if llm_provider in ["openai", "anthropic"]:
        config = ProcessorConfig.create_signature_preset(llm_provider, output_dir)
    else:
        # Use open-source preset
        preset_map = {
            "huggingface": "qwen-medium",
            "ollama": "ollama-llava"
        }
        preset = preset_map.get(llm_provider, "qwen-medium")
        config = ProcessorConfig.create_signature_preset(preset, output_dir)
        config.llm.provider = llm_provider
    
    if llm_model:
        config.llm.model = llm_model
    
    # Process signatures
    adapter = CSVSourceAdapter(config, csv_path)
    orchestrator = SignatureOrchestrator(config)
    
    sources = adapter.get_pdf_sources()
    for source in sources:
        source['pdf_path'] = adapter.prepare_source(source)
    
    return orchestrator.process_multiple_sources(sources)

def extract_signatures_from_directory(
    directory_path: str,
    output_dir: str = "signature_results",
    llm_provider: str = "openai",
    llm_model: str = None,
    recursive: bool = False,
    **kwargs
):
    """
    Simple interface for extracting signatures from directory.
    
    Args:
        directory_path: Path to directory containing PDFs
        output_dir: Output directory for results
        llm_provider: LLM provider to use
        llm_model: LLM model to use
        recursive: Scan subdirectories
        **kwargs: Additional configuration options
        
    Returns:
        List of extraction results
    """
    from .main import SignatureOrchestrator
    
    # Create configuration
    if llm_provider in ["openai", "anthropic"]:
        config = ProcessorConfig.create_signature_preset(llm_provider, output_dir)
    else:
        preset_map = {
            "huggingface": "qwen-medium",
            "ollama": "ollama-llava"
        }
        preset = preset_map.get(llm_provider, "qwen-medium")
        config = ProcessorConfig.create_signature_preset(preset, output_dir)
        config.llm.provider = llm_provider
    
    if llm_model:
        config.llm.model = llm_model
    
    config.source_config = {'recursive': recursive}
    
    # Process signatures
    adapter = DirectorySourceAdapter(config, directory_path)
    orchestrator = SignatureOrchestrator(config)
    
    sources = adapter.get_pdf_sources()
    for source in sources:
        source['pdf_path'] = adapter.prepare_source(source)
    
    return orchestrator.process_multiple_sources(sources)

__version__ = "2.0.0"
__author__ = "PDF Processor Team"

__all__ = [
    # Configuration classes
    "ProcessorConfig",
    "LLMConfig", 
    "ProcessingConfig",
    "TaskConfig",
    
    # Core components
    "DocumentProcessor",
    "SignatureAnalyzer", 
    "LLMInterface",
    
    # Factory functions
    "create_llm_interface",
    "get_available_providers",
    "get_recommended_provider",
    
    # Source adapters
    "CSVSourceAdapter",
    "DirectorySourceAdapter",
    "URLSourceAdapter",
    
    # Utilities
    "ProgressTracker",
    "setup_logging",
    "load_prompt",
    
    # Convenience functions
    "process_document",
    "extract_signatures_from_csv",
    "extract_signatures_from_directory"
]