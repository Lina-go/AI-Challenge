"""
Utility modules for the signature extraction framework.

This package contains focused utility modules, each handling a specific concern:

- pdf_converter: PDF to image conversion utilities
- llm_interface: LLM communication and prompt processing
- prompt_loader: Prompt template management and loading
- data_formatter: Data formatting, cleaning, and validation utilities
- progress_tracker: Progress monitoring and reporting
- logging_config: Logging configuration and setup

All utilities follow the Single Responsibility Principle and can be
used independently across the framework.
"""

# Core utilities - most commonly used
from .progress_tracker import ProgressTracker
from .logging_config import setup_logging, get_logger
from .prompt_loader import load_prompt, get_available_prompts

# Processing utilities
from .pdf_converter import (
    convert_pdf_to_images,
    save_page_image,
    get_pdf_info
)

from .llm_interface import (
    LLMInterface,
    create_llm_interface,
    get_available_providers
)

from .data_formatter import (
    format_signature_data,
    clean_extracted_text,
    standardize_date,
    validate_signature_result
)

__all__ = [
    # Core utilities
    "ProgressTracker",
    "setup_logging",
    "get_logger",
    "load_prompt",
    "get_available_prompts",
    
    # PDF processing
    "convert_pdf_to_images",
    "save_page_image", 
    "get_pdf_info",
    
    # LLM interface
    "LLMInterface",
    "create_llm_interface",
    "get_available_providers",
    
    # Data formatting
    "format_signature_data",
    "clean_extracted_text",
    "standardize_date",
    "validate_signature_result"
]

# Utility constants
SUPPORTED_IMAGE_FORMATS = ["PNG", "JPEG", "TIFF"]
DEFAULT_IMAGE_DPI = 300
MAX_IMAGE_SIZE = (4000, 4000)  # Max width, height in pixels

# LLM provider constants
SUPPORTED_LLM_PROVIDERS = ["openai", "anthropic", "azure"]
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4000

def get_utility_info():
    """Get information about available utilities."""
    return {
        "utilities": list(__all__),
        "supported_formats": {
            "images": SUPPORTED_IMAGE_FORMATS,
            "llm_providers": SUPPORTED_LLM_PROVIDERS
        },
        "defaults": {
            "image_dpi": DEFAULT_IMAGE_DPI,
            "llm_temperature": DEFAULT_LLM_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "max_image_size": MAX_IMAGE_SIZE
        }
    }

def validate_environment():
    """Validate that required dependencies are available."""
    validation_results = {
        "pdf_processing": True,
        "image_processing": True, 
        "llm_providers": {}
    }
    
    # Check PDF processing
    try:
        import fitz  # PyMuPDF
    except ImportError:
        validation_results["pdf_processing"] = False
    
    # Check image processing
    try:
        from PIL import Image
    except ImportError:
        validation_results["image_processing"] = False
    
    # Check LLM providers
    for provider in SUPPORTED_LLM_PROVIDERS:
        try:
            if provider == "openai":
                import openai
                validation_results["llm_providers"][provider] = True
            elif provider == "anthropic":
                import anthropic
                validation_results["llm_providers"][provider] = True
            else:
                validation_results["llm_providers"][provider] = False
        except ImportError:
            validation_results["llm_providers"][provider] = False
    
    return validation_results