# signature_extractor_v5/adapters/directory_adapter.py
"""Directory source adapter."""

import logging
from pathlib import Path
from typing import List, Dict, Any

from .base import BaseSourceAdapter

logger = logging.getLogger(__name__)


class DirectorySourceAdapter(BaseSourceAdapter):
    """Handles directory of PDF files."""
    
    def __init__(self, directory_path: str):
        self.directory_path = Path(directory_path)
        
        if not self.directory_path.exists():
            raise ValueError(f"Directory not found: {directory_path}")
    
    def get_sources(self) -> List[Dict[str, Any]]:
        """Scan directory for PDFs."""
        pdf_files = list(self.directory_path.glob("*.pdf"))
        
        sources = []
        for idx, pdf_file in enumerate(sorted(pdf_files)):
            source = {
                "source_id": f"dir_{idx}_{pdf_file.stem}",
                "pdf_path": str(pdf_file),
                "document_name": pdf_file.stem
            }
            sources.append(source)
        
        logger.info(f"Found {len(sources)} PDFs in directory")
        return sources
    
    def prepare_source(self, source: Dict[str, Any]) -> str:
        """Return path (already local)."""
        return source["pdf_path"]