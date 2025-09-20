import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from ..base_extractor import BaseSourceAdapter
from ..config import ExtractionConfig

logger = logging.getLogger(__name__)

class DirectorySourceAdapter(BaseSourceAdapter):
    """Adapter for processing PDFs from a directory"""
    
    def __init__(self, config: ExtractionConfig, directory_path: str):
        super().__init__(config)
        self.directory_path = Path(directory_path)
        
        if not self.directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        if not self.directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
    
    def get_pdf_sources(self) -> List[Dict[str, Any]]:
        """Scan directory for PDF files"""
        try:
            # Find all PDF files
            pdf_files = list(self.directory_path.glob("*.pdf"))
            
            # Check subdirectories if recursive processing is enabled
            if self.config.source_config and self.config.source_config.get('recursive', False):
                pdf_files.extend(list(self.directory_path.rglob("*.pdf")))
            
            # Remove duplicates and sort
            pdf_files = sorted(list(set(pdf_files)))
            
            sources = []
            for idx, pdf_file in enumerate(pdf_files):
                source = {
                    'source_id': f"dir_{idx}_{pdf_file.stem}",
                    'pdf_path': str(pdf_file),
                    'document_name': pdf_file.stem,
                    'file_size': pdf_file.stat().st_size,
                    'relative_path': str(pdf_file.relative_to(self.directory_path))
                }
                sources.append(source)
            
            logger.info(f"Found {len(sources)} PDF files in directory: {self.directory_path}")
            return sources
            
        except Exception as e:
            logger.error(f"Error scanning directory {self.directory_path}: {e}")
            raise
    
    def prepare_source(self, source: Dict[str, Any]) -> str:
        """Return the local path (already local)"""
        pdf_path = source['pdf_path']
        
        # Verify file still exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        return pdf_path