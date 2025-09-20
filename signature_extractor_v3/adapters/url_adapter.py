import os
import logging
import requests
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlparse
from ..base_extractor import BaseSourceAdapter
from ..config import ExtractionConfig

logger = logging.getLogger(__name__)

class URLSourceAdapter(BaseSourceAdapter):
    """Adapter for processing PDFs from a list of URLs"""
    
    def __init__(self, config: ExtractionConfig, url_list_path: str):
        super().__init__(config)
        self.url_list_path = url_list_path
        self.download_dir = Path(config.processing.output_dir) / "downloads"
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def get_pdf_sources(self) -> List[Dict[str, Any]]:
        """Read URL list file and return list of PDF sources"""
        try:
            with open(self.url_list_path, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            # Filter out empty lines and comments
            urls = [url for url in urls if url and not url.startswith('#')]
            
            sources = []
            for idx, url in enumerate(urls):
                # Parse URL to get filename
                parsed_url = urlparse(url)
                filename = Path(parsed_url.path).name
                if not filename.endswith('.pdf'):
                    filename = f"document_{idx}.pdf"
                
                source = {
                    'source_id': f"url_{idx}_{Path(filename).stem}",
                    'pdf_url': url,
                    'document_name': Path(filename).stem,
                    'original_index': idx,
                    'expected_filename': filename
                }
                sources.append(source)
            
            logger.info(f"Found {len(sources)} PDF URLs in list")
            return sources
            
        except Exception as e:
            logger.error(f"Error reading URL list {self.url_list_path}: {e}")
            raise
    
    def prepare_source(self, source: Dict[str, Any]) -> str:
        """Download PDF from URL"""
        pdf_url = source['pdf_url']
        
        try:
            # Generate local filename
            filename = source['expected_filename']
            local_path = self.download_dir / filename
            
            # Skip if already downloaded
            if local_path.exists():
                logger.debug(f"PDF already downloaded: {local_path}")
                return str(local_path)
            
            # Download PDF with progress logging
            logger.info(f"Downloading PDF: {pdf_url}")
            
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(
                pdf_url, 
                timeout=self.config.processing.timeout,
                headers=headers,
                stream=True
            )
            response.raise_for_status()
            
            # Check if response is actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type:
                logger.warning(f"URL might not be a PDF: {pdf_url} (Content-Type: {content_type})")
            
            # Save PDF with progress tracking
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Log progress for large files
                        if total_size > 0 and downloaded_size % (1024 * 1024) == 0:  # Every MB
                            progress = (downloaded_size / total_size) * 100
                            logger.debug(f"Download progress: {progress:.1f}%")
            
            # Verify file was downloaded successfully
            if not local_path.exists() or local_path.stat().st_size == 0:
                raise Exception("Downloaded file is empty or doesn't exist")
            
            logger.info(f"Successfully downloaded: {local_path} ({local_path.stat().st_size} bytes)")
            return str(local_path)
            
        except requests.RequestException as e:
            logger.error(f"HTTP error downloading {pdf_url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error preparing source {pdf_url}: {e}")
            raise
    
    def validate_url_list(self) -> List[str]:
        """
        Validate URLs in the list without downloading
        
        Returns:
            List of validation errors (empty if all URLs are valid)
        """
        errors = []
        
        try:
            with open(self.url_list_path, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            for idx, url in enumerate(urls):
                try:
                    parsed_url = urlparse(url)
                    if not parsed_url.scheme or not parsed_url.netloc:
                        errors.append(f"Line {idx + 1}: Invalid URL format: {url}")
                except Exception as e:
                    errors.append(f"Line {idx + 1}: Error parsing URL {url}: {e}")
            
        except Exception as e:
            errors.append(f"Error reading URL list file: {e}")
        
        return errors