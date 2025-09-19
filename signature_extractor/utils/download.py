import os
import requests
import urllib.parse
import hashlib
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PDFDownloader:
    """
    Handles PDF downloading with caching and error handling.
    
    Single Responsibility: PDF download management.
    """
    
    def __init__(self, config, download_dir: str = "downloaded_pdfs"):
        self.config = config
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
    
    def download_pdf(self, url: str) -> Optional[str]:
        """
        Download PDF from URL with caching.
        
        Args:
            url: PDF URL to download
            
        Returns:
            Local path to downloaded PDF, None if failed
        """
        try:
            # Generate filename from URL
            filename = self._generate_filename(url)
            local_path = self.download_dir / filename
            
            # Skip if already exists
            if local_path.exists():
                logger.info(f"File already exists: {filename}")
                return str(local_path)
            
            # Download PDF
            logger.info(f"Downloading: {filename}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save PDF
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Downloaded successfully: {filename}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None
    
    def _generate_filename(self, url: str) -> str:
        """Generate a filename from URL."""
        parsed_url = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename or not filename.endswith('.pdf'):
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"document_{url_hash}.pdf"
        
        return filename
