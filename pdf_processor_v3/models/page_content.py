"""
Data model for extracted page content.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class PageContent:
    """
    Container for all content extracted from a single page.
    
    Attributes:
        page_number: The page number (1-based)
        text_content: Extracted text with placeholders for tables/figures/signatures
        table_content: Raw table extraction results
        figure_content: Raw figure extraction results
        signature_content: Raw signature extraction results (not used for replacement)
        table_mapping: Parsed mapping of table IDs to markdown content
        figure_mapping: Parsed mapping of figure IDs to markdown content
        signature_mapping: Parsed mapping of signature IDs (signatures remain as placeholders)
        integrated_content: Final integrated markdown content
    """
    page_number: int
    text_content: Optional[str] = None
    table_content: Optional[str] = None
    figure_content: Optional[str] = None
    signature_content: Optional[str] = None
    table_mapping: Dict[str, str] = field(default_factory=dict)
    figure_mapping: Dict[str, str] = field(default_factory=dict)
    signature_mapping: Dict[str, str] = field(default_factory=dict)
    integrated_content: Optional[str] = None
    
    def has_content(self) -> bool:
        """Check if this page has any extracted content."""
        return bool(self.text_content or self.table_content or self.figure_content or self.signature_content)
    
    def get_placeholder_count(self) -> Dict[str, int]:
        """Count placeholders in text content."""
        if not self.text_content:
            return {"tables": 0, "figures": 0, "signatures": 0}
        
        table_count = self.text_content.count("[TABLE_")
        figure_count = self.text_content.count("[FIGURE_")
        signature_count = self.text_content.count("[SIGNATURE_")
        
        return {"tables": table_count, "figures": figure_count, "signatures": signature_count}
