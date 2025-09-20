# signature_extractor_v5/core/document_processor.py
"""Document processing and page extraction."""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles PDF to image conversion."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "page_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_pages_as_images(
        self, 
        pdf_path: str,
        dpi: int = 300,
        max_pages: Optional[int] = None
    ) -> List[Tuple[int, Image.Image]]:
        """
        Extract PDF pages as PIL Images.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image extraction
            max_pages: Maximum pages to process
            
        Returns:
            List of (page_number, PIL.Image) tuples
        """
        pages = []
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = min(doc.page_count, max_pages) if max_pages else doc.page_count
            
            logger.info(f"Processing {total_pages} pages from {pdf_path}")
            
            for page_num in range(total_pages):
                page = doc[page_num]
                
                # Convert to image
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                pages.append((page_num + 1, img))
                
                logger.debug(f"Extracted page {page_num + 1}")
            
            doc.close()
            logger.info(f"Successfully extracted {len(pages)} pages")
            
        except Exception as e:
            logger.error(f"Error extracting pages: {e}")
            raise
        
        return pages
    
    def save_page_image(self, image: Image.Image, pdf_name: str, page_num: int) -> str:
        """
        Save page image to disk.
        
        Args:
            image: PIL Image
            pdf_name: Name of source PDF
            page_num: Page number
            
        Returns:
            Path to saved image
        """
        filename = f"{pdf_name}_page_{page_num}.png"
        filepath = self.images_dir / filename
        image.save(filepath)
        return str(filepath)