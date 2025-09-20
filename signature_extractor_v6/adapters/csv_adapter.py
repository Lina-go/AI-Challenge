import os
import logging
import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlparse
from ..base_extractor import BaseSourceAdapter
from ..config import ExtractionConfig

logger = logging.getLogger(__name__)

class CSVSourceAdapter(BaseSourceAdapter):
    """Adapter for processing PDFs listed in CSV files"""
    
    def __init__(self, config: ExtractionConfig, csv_path: str):
        super().__init__(config)
        self.csv_path = csv_path
        self.download_dir = Path(config.processing.output_dir) / "downloads"
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def get_pdf_sources(self) -> List[Dict[str, Any]]:
        """Read CSV and return list of PDFs"""
        try:
            df = pd.read_csv(self.csv_path)
            
            # Expected columns: pdf_url, document_name
            if 'pdf_url' not in df.columns:
                raise ValueError("CSV must contain 'pdf_url' column")
            
            sources = []
            for idx, row in df.iterrows():
                source = {
                    'source_id': f"csv_{idx}",
                    'pdf_url': row['pdf_url'],
                    'document_name': row.get('document_name', f"document_{idx}"),
                    'original_index': idx
                }
                sources.append(source)
            
            logger.info(f"Found {len(sources)} PDF sources in CSV")
            return sources
            
        except Exception as e:
            logger.error(f"Error reading CSV {self.csv_path}: {e}")
            raise
    
    def prepare_source(self, source: Dict[str, Any]) -> str:
        """Download PDF if URL, or return local path"""
        pdf_url = source['pdf_url']
        
        # Check if it's already a local file
        if os.path.exists(pdf_url):
            logger.debug(f"Using local file: {pdf_url}")
            return pdf_url
        
        # Download from URL
        try:
            parsed_url = urlparse(pdf_url)
            if not parsed_url.scheme:
                raise ValueError(f"Invalid URL: {pdf_url}")
            
            # Generate local filename
            filename = f"{source['source_id']}.pdf"
            local_path = self.download_dir / filename
            
            # Skip if already downloaded
            if local_path.exists():
                logger.debug(f"PDF already downloaded: {local_path}")
                return str(local_path)
            
            # Download PDF
            logger.info(f"Downloading PDF: {pdf_url}")
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # Save PDF
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded to: {local_path}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Error preparing source {pdf_url}: {e}")
            raise