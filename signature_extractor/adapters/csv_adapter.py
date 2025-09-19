import polars as pl
from ..base_extractor import BaseSourceAdapter
from ..utils.download import PDFDownloader
from typing import List, Dict, Any
from ..config import ExtractorConfig

class CSVSourceAdapter(BaseSourceAdapter):
    """
    Adapter for processing PDFs listed in a CSV file.
    
    Handles the specific logic for CSV-based PDF sources.
    """
    
    def __init__(self, config: ExtractorConfig, csv_path: str):
        super().__init__(config)
        self.csv_path = csv_path
        self.downloader = PDFDownloader(config)
    
    def get_pdf_sources(self) -> List[Dict[str, Any]]:
        """Load PDF sources from CSV."""
        df = pl.read_csv(self.csv_path)
        
        sources = []
        for i, row in enumerate(df.iter_rows(named=True)):
            sources.append({
                'source_id': f"pdf_{i:03d}",
                'pdf_url': row.get('pdf_url'),
                'metadata': dict(row)
            })
        
        return sources
    
    def prepare_source(self, source: Dict[str, Any]) -> str:
        """Download PDF from URL if needed."""
        if 'pdf_url' in source:
            return self.downloader.download_pdf(source['pdf_url'])
        else:
            raise ValueError(f"No PDF URL found in source: {source['source_id']}")