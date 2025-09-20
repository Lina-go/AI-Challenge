import logging
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes PDF documents and converts them to images"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.processing.output_dir)
        self.images_dir = self.output_dir / "page_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def process_document(self, pdf_path: str) -> List[Tuple[int, str]]:
        """
        Convert PDF to page images
        
        Returns:
            List of tuples (page_number, image_path)
        """
        try:
            doc = fitz.open(pdf_path)
            page_images = []
            
            logger.info(f"Processing PDF with {doc.page_count} pages: {pdf_path}")
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(self.config.processing.dpi / 72, self.config.processing.dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                
                # Save as PNG
                pdf_name = Path(pdf_path).stem
                image_filename = f"{pdf_name}_page_{page_num + 1}.png"
                image_path = self.images_dir / image_filename
                
                pix.save(str(image_path))
                page_images.append((page_num + 1, str(image_path)))
                
                logger.debug(f"Converted page {page_num + 1} to {image_path}")
            
            doc.close()
            logger.info(f"Successfully processed {len(page_images)} pages")
            return page_images
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def extract_page_text(self, pdf_path: str, page_num: int) -> str:
        """Extract text from specific PDF page"""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]  # Convert to 0-based index
            text = page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from page {page_num}: {e}")
            return ""
    
    def get_pdf_metadata(self, pdf_path: str) -> dict:
        """Extract PDF metadata"""
        try:
            doc = fitz.open(pdf_path)
            metadata = {
                'page_count': doc.page_count,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', '')
            }
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {}