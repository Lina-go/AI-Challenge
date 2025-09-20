"""
Source adapters for different PDF input types.

This package provides adapters for various PDF sources:

- CSVSourceAdapter: Process PDFs listed in CSV files with URLs
- DirectorySourceAdapter: Process all PDFs in a local directory  
- URLSourceAdapter: Process PDFs from a list of URLs

Each adapter implements the BaseSourceAdapter interface and handles:
- Source discovery and enumeration
- PDF acquisition (download if remote)
- Metadata extraction and preparation
- Error handling and validation

Adapters can be used independently or through the orchestration layer.
"""

from .csv_adapter import CSVSourceAdapter
from .directory_adapter import DirectorySourceAdapter
from .url_adapter import URLSourceAdapter

__all__ = [
    "CSVSourceAdapter",
    "DirectorySourceAdapter", 
    "URLSourceAdapter"
]

# Adapter registry for dynamic loading
ADAPTER_REGISTRY = {
    "csv": CSVSourceAdapter,
    "directory": DirectorySourceAdapter,
    "urls": URLSourceAdapter,
    "url_list": URLSourceAdapter  # Alias
}

# Supported source types
SUPPORTED_SOURCE_TYPES = list(ADAPTER_REGISTRY.keys())

# Default configurations for each adapter type
ADAPTER_DEFAULTS = {
    "csv": {
        "url_column": "pdf_url",
        "id_column": None,  # Auto-generate if not specified
        "metadata_columns": None  # Include all columns as metadata
    },
    "directory": {
        "recursive": False,
        "file_pattern": "*.pdf",
        "include_subdirs": False
    },
    "urls": {
        "url_format": "one_per_line",
        "skip_empty_lines": True,
        "comment_prefix": "#"
    }
}

def create_adapter(source_type: str, config, source_path: str, **kwargs):
    """
    Factory function to create appropriate source adapter.
    
    Args:
        source_type: Type of source ("csv", "directory", "urls")
        config: ExtractionConfig instance
        source_path: Path to source (file or directory)
        **kwargs: Additional adapter-specific arguments
        
    Returns:
        Configured source adapter instance
        
    Raises:
        ValueError: If source_type is not supported
    """
    if source_type not in ADAPTER_REGISTRY:
        supported = ", ".join(SUPPORTED_SOURCE_TYPES)
        raise ValueError(f"Unsupported source type: {source_type}. Supported: {supported}")
    
    adapter_class = ADAPTER_REGISTRY[source_type]
    
    # Merge default configuration with kwargs
    adapter_config = ADAPTER_DEFAULTS.get(source_type, {}).copy()
    adapter_config.update(kwargs)
    
    return adapter_class(config, source_path, **adapter_config)

def get_supported_source_types():
    """Get list of supported source types."""
    return SUPPORTED_SOURCE_TYPES.copy()

def get_adapter_defaults(source_type: str = None):
    """
    Get default configuration for adapter type(s).
    
    Args:
        source_type: Specific adapter type, or None for all
        
    Returns:
        Default configuration dictionary
    """
    if source_type:
        return ADAPTER_DEFAULTS.get(source_type, {}).copy()
    return ADAPTER_DEFAULTS.copy()

def validate_source(source_type: str, source_path: str) -> dict:
    """
    Validate that a source is accessible and properly formatted.
    
    Args:
        source_type: Type of source to validate
        source_path: Path to source
        
    Returns:
        Validation result dictionary with status and details
    """
    from pathlib import Path
    
    result = {
        "valid": False,
        "source_type": source_type,
        "source_path": source_path,
        "issues": []
    }
    
    if source_type not in SUPPORTED_SOURCE_TYPES:
        result["issues"].append(f"Unsupported source type: {source_type}")
        return result
    
    path = Path(source_path)
    
    if source_type == "csv":
        if not path.exists():
            result["issues"].append("CSV file does not exist")
        elif not path.suffix.lower() == ".csv":
            result["issues"].append("File is not a CSV file")
        else:
            # Additional CSV validation could go here
            result["valid"] = True
            
    elif source_type == "directory":
        if not path.exists():
            result["issues"].append("Directory does not exist")
        elif not path.is_dir():
            result["issues"].append("Path is not a directory")
        else:
            # Check for PDF files
            pdf_files = list(path.glob("*.pdf"))
            if not pdf_files:
                result["issues"].append("No PDF files found in directory")
            else:
                result["valid"] = True
                result["pdf_count"] = len(pdf_files)
                
    elif source_type in ["urls", "url_list"]:
        if not path.exists():
            result["issues"].append("URL file does not exist")
        else:
            result["valid"] = True
    
    return result