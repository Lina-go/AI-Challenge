"""
PDF to image conversion utilities.
"""

import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Tuple
from pathlib import Path


def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Tuple[int, Image.Image]]:
    """
    Convert PDF pages to high-resolution images for LLM processing.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for image conversion (default: 300)
        
    Returns:
        List of tuples containing (page_number, PIL_Image)
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF cannot be processed
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(str(pdf_path))
        page_images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Convert to image with high DPI for better text recognition
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Store as (1-based page number, image)
            page_images.append((page_num + 1, image))
        
        doc.close()
        return page_images
        
    except Exception as e:
        raise Exception(f"Failed to convert PDF to images: {e}")


def save_page_image(image: Image.Image, output_path: str) -> None:
    """
    Save a page image to disk for debugging.
    
    Args:
        image: PIL Image to save
        output_path: Path where to save the image
    """
    try:
        image.save(output_path, "PNG")
    except Exception as e:
        raise Exception(f"Failed to save image to {output_path}: {e}")


def get_pdf_info(pdf_path: str) -> dict:
    """
    Get basic information about a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with PDF metadata
    """
    try:
        doc = fitz.open(pdf_path)
        info = {
            "page_count": len(doc),
            "metadata": doc.metadata,
            "needs_password": doc.needs_pass,
            "is_pdf": doc.is_pdf
        }
        doc.close()
        return info
    except Exception as e:
        raise Exception(f"Failed to get PDF info: {e}")
