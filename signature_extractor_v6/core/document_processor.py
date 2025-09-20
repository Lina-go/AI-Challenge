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
        Convert PDF to page images with optimization for HuggingFace
        
        Returns:
            List of tuples (page_number, image_path)
        """
        try:
            doc = fitz.open(pdf_path)
            page_images = []
            
            logger.info(f"Processing PDF with {doc.page_count} pages: {pdf_path}")
            
            # Detect if using HuggingFace provider and adjust settings
            is_huggingface = (hasattr(self.config, 'llm') and 
                            hasattr(self.config.llm, 'provider') and 
                            self.config.llm.provider.lower() == "huggingface")
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Adjust DPI based on provider
                if is_huggingface:
                    # Lower DPI for HuggingFace to reduce payload size
                    dpi = min(self.config.processing.dpi, 200)
                else:
                    dpi = self.config.processing.dpi
                
                # Convert page to image
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                
                # Save as PNG
                pdf_name = Path(pdf_path).stem
                image_filename = f"{pdf_name}_page_{page_num + 1}.png"
                image_path = self.images_dir / image_filename
                
                pix.save(str(image_path))
                
                # Additional optimization for HuggingFace
                if is_huggingface:
                    self._optimize_image_for_hf(str(image_path))
                
                page_images.append((page_num + 1, str(image_path)))
                
                logger.debug(f"Converted page {page_num + 1} to {image_path}")
            
            doc.close()
            logger.info(f"Successfully processed {len(page_images)} pages")
            return page_images
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def _optimize_image_for_hf(self, image_path: str) -> None:
        """
        Optimize image size for HuggingFace Router limits
        """
        try:
            with Image.open(image_path) as img:
                # Get current size
                width, height = img.size
                file_size = Path(image_path).stat().st_size
                
                # If image is too large, resize it
                max_dimension = 750  # Max width or height
                max_file_size = 1 * 1024 * 1024  # 2MB limit
                
                if width > max_dimension or height > max_dimension or file_size > max_file_size:
                    logger.debug(f"Optimizing image {image_path} (current: {width}x{height}, {file_size/1024:.0f}KB)")
                    
                    # Calculate new dimensions maintaining aspect ratio
                    if width > height:
                        new_width = min(width, max_dimension)
                        new_height = int((new_width / width) * height)
                    else:
                        new_height = min(height, max_dimension)
                        new_width = int((new_height / height) * width)
                    
                    # Resize image
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save with compression
                    resized_img.save(image_path, "PNG", optimize=True)
                    
                    new_file_size = Path(image_path).stat().st_size
                    logger.debug(f"Optimized to {new_width}x{new_height}, {new_file_size/1024:.0f}KB")
                    
        except Exception as e:
            logger.warning(f"Failed to optimize image {image_path}: {e}")
    
    def extract_page_text(self, pdf_path: str, page_num: int) -> str:
        """Extract text from specific PDF page"""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]
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