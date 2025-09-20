"""
Data models and structures for signature extraction.

This package contains data models that represent the various entities
and results in the signature extraction process:

- ExtractionResult: Final results from processing PDFs
- SignatureData: Individual signature information and metadata
- DocumentMetadata: Document-level information and properties

All models use dataclasses for clean, type-safe data structures
with automatic serialization support.
"""

from .extraction_result import (
    ExtractionResult,
    SourceResult,
    ProcessingSummary
)

from .signature_data import (
    SignatureData,
    SignatureMetadata,
    SignatoryInfo,
    DateInfo
)

from .document_metadata import (
    DocumentMetadata,
    PageInfo,
    ProcessingStats
)

__all__ = [
    # Extraction results
    "ExtractionResult",
    "SourceResult", 
    "ProcessingSummary",
    
    # Signature data
    "SignatureData",
    "SignatureMetadata",
    "SignatoryInfo",
    "DateInfo",
    
    # Document metadata
    "DocumentMetadata",
    "PageInfo",
    "ProcessingStats"
]

# Model version for compatibility tracking
MODEL_VERSION = "2.0.0"

# Field mappings for output format compatibility
SIGNATURE_COLUMNS_MAPPING = {
    "Signature": "has_signature",
    "Signature_Yes_text": "signatory_text", 
    "Signature_Image": "is_image_signature",
    "Signature_scanned": "is_scanned",
    "Presence_Signature_date": "has_date",
    "Signature_Date": "signature_date"
}

def create_signature_result_dict(signature_data: 'SignatureData') -> dict:
    """
    Convert SignatureData to dictionary matching signature_columns.xlsx format.
    
    Args:
        signature_data: SignatureData instance
        
    Returns:
        Dictionary with keys matching signature_columns.xlsx
    """
    return {
        "Signature": "Yes" if signature_data.has_signature else "No",
        "Signature_Yes_text": signature_data.signatory_text or "",
        "Signature_Image": "Yes" if signature_data.is_image_signature else "No", 
        "Signature_scanned": "Yes" if signature_data.is_scanned else "No",
        "Presence_Signature_date": "Yes" if signature_data.has_date else "No",
        "Signature_Date": signature_data.signature_date or ""
    }

def validate_signature_result(result_dict: dict) -> tuple[bool, list]:
    """
    Validate that result dictionary has required signature columns.
    
    Args:
        result_dict: Dictionary to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    required_columns = list(SIGNATURE_COLUMNS_MAPPING.keys())
    issues = []
    
    for column in required_columns:
        if column not in result_dict:
            issues.append(f"Missing required column: {column}")
    
    # Validate value formats
    if "Signature" in result_dict and result_dict["Signature"] not in ["Yes", "No"]:
        issues.append("Signature must be 'Yes' or 'No'")
    
    if "Signature_Image" in result_dict and result_dict["Signature_Image"] not in ["Yes", "No"]:
        issues.append("Signature_Image must be 'Yes' or 'No'")
    
    if "Signature_scanned" in result_dict and result_dict["Signature_scanned"] not in ["Yes", "No"]:
        issues.append("Signature_scanned must be 'Yes' or 'No'")
    
    if "Presence_Signature_date" in result_dict and result_dict["Presence_Signature_date"] not in ["Yes", "No"]:
        issues.append("Presence_Signature_date must be 'Yes' or 'No'")
    
    return len(issues) == 0, issues

def get_model_info():
    """Get information about available data models."""
    return {
        "version": MODEL_VERSION,
        "models": list(__all__),
        "output_format": {
            "columns": list(SIGNATURE_COLUMNS_MAPPING.keys()),
            "mapping": SIGNATURE_COLUMNS_MAPPING
        }
    }

def create_empty_signature_result() -> dict:
    """Create an empty signature result with all required columns."""
    return {
        "Signature": "No",
        "Signature_Yes_text": "",
        "Signature_Image": "No",
        "Signature_scanned": "No", 
        "Presence_Signature_date": "No",
        "Signature_Date": ""
    }

# Utility functions for data conversion
def results_to_dataframe(results: list) -> 'pd.DataFrame':
    """
    Convert list of signature results to pandas DataFrame.
    
    Args:
        results: List of signature result dictionaries
        
    Returns:
        pandas DataFrame with signature data
    """
    try:
        import pandas as pd
        return pd.DataFrame(results)
    except ImportError:
        raise ImportError("pandas required for DataFrame conversion")

def save_results_to_csv(results: list, output_path: str):
    """
    Save signature results to CSV file.
    
    Args:
        results: List of signature result dictionaries  
        output_path: Path where to save CSV file
    """
    df = results_to_dataframe(results)
    df.to_csv(output_path, index=False)

def save_results_to_json(results: list, output_path: str):
    """
    Save signature results to JSON file.
    
    Args:
        results: List of signature result dictionaries
        output_path: Path where to save JSON file  
    """
    import json
    from pathlib import Path
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)