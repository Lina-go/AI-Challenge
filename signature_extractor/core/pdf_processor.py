import fitz  # PyMuPDF
import os
from typing import List, Tuple
from pathlib import Path
from ..config import ExtractorConfig

class PDFProcessor:
    """
    Handles PDF to image conversion
    """
    
    def __init__(self, config: ExtractorConfig):
        self.config = config
    
    def pdf_to_images(self, pdf_path: str) -> List[Tuple[int, str]]:
        """
        Convert PDF pages to PNG images
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of (page_number, image_path) tuples
        """
        output_dir = self._get_images_dir(pdf_path)
        os.makedirs(output_dir, exist_ok=True)
        
        doc = fitz.open(pdf_path)
        page_images = []
        
        for i, page in enumerate(doc, 1):
            pix = page.get_pixmap()
            image_path = os.path.join(output_dir, f"page_{i:03d}.png")
            pix.save(image_path)
            page_images.append((i, image_path))
        
        doc.close()
        return page_images
    
    def _get_images_dir(self, pdf_path: str) -> str:
        """Generate output directory for page images"""
        pdf_name = Path(pdf_path).stem
        return os.path.join(
            self.config.processing.output_dir,
            f"{pdf_name}_pages"
        )