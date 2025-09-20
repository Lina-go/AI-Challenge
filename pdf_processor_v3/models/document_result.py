"""
Data model for final document processing results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .page_content import PageContent


@dataclass
class DocumentResult:
    """
    Final results from processing an entire PDF document.
    
    Attributes:
        pdf_path: Path to the original PDF file
        total_pages: Total number of pages in the PDF
        processed_pages: Number of pages that were processed
        skipped_pages: Number of pages that were skipped
        page_contents: List of PageContent objects for processed pages
        final_markdown: Final integrated markdown document
        processing_stats: Statistics about the processing
    """
    pdf_path: str
    total_pages: int
    processed_pages: int = 0
    skipped_pages: int = 0
    page_contents: List[PageContent] = field(default_factory=list)
    final_markdown: Optional[str] = None
    processing_stats: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize processing stats."""
        if not self.processing_stats:
            self.processing_stats = {
                "total_tables": 0,
                "total_figures": 0,
                "total_placeholders": 0,
                "successful_integrations": 0
            }
    
    def add_page_content(self, page_content: PageContent) -> None:
        """Add processed page content and update stats."""
        self.page_contents.append(page_content)
        self.processed_pages += 1
        
        # Update stats
        placeholder_counts = page_content.get_placeholder_count()
        self.processing_stats["total_tables"] += len(page_content.table_mapping)
        self.processing_stats["total_figures"] += len(page_content.figure_mapping)
        self.processing_stats["total_placeholders"] += (
            placeholder_counts["tables"] + placeholder_counts["figures"]
        )
        
        if page_content.integrated_content:
            self.processing_stats["successful_integrations"] += 1
    
    def get_success_rate(self) -> float:
        """Calculate processing success rate."""
        if self.total_pages == 0:
            return 0.0
        return (self.processed_pages / self.total_pages) * 100
