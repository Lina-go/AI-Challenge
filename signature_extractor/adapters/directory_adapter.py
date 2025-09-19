from typing import List, Dict, Any
import glob
from pathlib import Path
from ..base_extractor import BaseSourceAdapter
from ..config import ExtractorConfig


class DirectorySourceAdapter(BaseSourceAdapter):
    """
    Adapter for processing PDFs from a directory.
    """
    
    def __init__(self, config: ExtractorConfig, directory_path: str):
        super().__init__(config)
        self.directory_path = Path(directory_path)
        
        if not self.directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        if not self.directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
    
    def get_pdf_sources(self) -> List[Dict[str, Any]]:
        """Find all PDFs in directory."""
        pdf_files = list(self.directory_path.glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in: {self.directory_path}")
        
        sources = []
        for i, pdf_path in enumerate(pdf_files):
            pdf_name = pdf_path.stem
            sources.append({
                'source_id': f"pdf_{i:03d}_{pdf_name}",
                'pdf_path': str(pdf_path),
                'metadata': {
                    'filename': pdf_path.name,
                    'size_bytes': pdf_path.stat().st_size
                }
            })
        
        return sources
    
    def prepare_source(self, source: Dict[str, Any]) -> str:
        """PDF is already local, just return path."""
        return source['pdf_path']