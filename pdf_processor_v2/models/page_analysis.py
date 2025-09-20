"""
Data model for page content analysis results.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PageAnalysis:
    """
    Results from analyzing a page to determine content types and processing needs.
    
    Attributes:
        page_number: The page number (1-based)
        should_parse: Whether this page contains meaningful content worth processing
        has_text: Whether the page contains narrative text content
        has_tables: Whether the page contains structured tabular data
        has_figures: Whether the page contains visual elements (charts, diagrams)
        has_signatures: Whether the page contains signature elements
        raw_analysis: Optional raw response from the analysis prompt
    """
    page_number: int
    should_parse: bool
    has_text: bool
    has_tables: bool
    has_figures: bool
    has_signatures: bool
    raw_analysis: Optional[str] = None
    
    def needs_processing(self) -> bool:
        """Check if this page needs any content extraction processing."""
        return self.should_parse and (self.has_text or self.has_tables or self.has_figures or self.has_signatures)
    
    def get_required_extractors(self) -> list[str]:
        """Get list of extractor types needed for this page."""
        extractors = []
        if self.has_text:
            extractors.append("text")
        if self.has_tables:
            extractors.append("tables")
        if self.has_figures:
            extractors.append("figures")
        if self.has_signatures:
            extractors.append("signatures")
        return extractors
