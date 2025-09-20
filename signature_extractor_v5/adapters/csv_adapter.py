# signature_extractor_v5/adapters/csv_adapter.py
"""CSV source adapter."""

import logging
import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict, Any

from .base import BaseSourceAdapter

logger = logging.getLogger(__name__)


class CSVSourceAdapter(BaseSourceAdapter):
    """Handles CSV files with PDF URLs."""
    
    def __init__(self, csv_path: str, download_dir: str = "downloads"):
        self.csv_path = csv_path
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def get_sources(self) -> List[Dict[str, Any]]:
        """Read CSV and return sources."""
        df = pd.read_csv(self.csv_path)
        
        if "pdf_url" not in df.columns:
            raise ValueError("CSV must contain 'pdf_url' column")
        
        sources = []
        for idx, row in df.iterrows():
            source = {
                "source_id": f"csv_{idx}",
                "pdf_url": row["pdf_url"],
                "document_name": row.get("document_name", f"doc_{idx}")
            }
            sources.append(source)
        
        logger.info(f"Found {len(sources)} sources in CSV")
        return sources
    
    def prepare_source(self, source: Dict[str, Any]) -> str:
        """Download PDF if needed."""
        pdf_url = source["pdf_url"]
        
        # Check if local file
        if Path(pdf_url).exists():
            source["pdf_path"] = pdf_url
            return pdf_url
        
        # Download from URL
        filename = f"{source['source_id']}.pdf"
        local_path = self.download_dir / filename
        
        if not local_path.exists():
            logger.info(f"Downloading: {pdf_url}")
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            
            with open(local_path, "wb") as f:
                f.write(response.content)
        
        source["pdf_path"] = str(local_path)
        return str(local_path)