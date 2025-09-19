from PIL import Image
from typing import Dict, Any
import os


class ImageProcessor:
    """
    Handles image cropping and manipulation.
    
    Single Responsibility: Image processing operations.
    """
    
    def __init__(self, config):
        self.config = config
    
    def save_signature_crop(
        self, 
        image_path: str, 
        detection: Dict[str, Any],
        source_id: str,
        page_number: int
    ) -> str:
        """
        Save cropped signature from detection.
        
        Args:
            image_path: Path to source image
            detection: Detection dictionary with bounding box
            source_id: Unique source identifier
            page_number: Page number
            
        Returns:
            Path to saved crop
        """
        if not self.config.processing.save_crops:
            return ""
        
        # Create crops directory
        crops_dir = os.path.join(
            self.config.processing.output_dir, 
            f"{source_id}_crops"
        )
        os.makedirs(crops_dir, exist_ok=True)
        
        # Generate crop filename
        crop_filename = f"p{page_number:03d}_sig_{detection['class_id']:02d}.png"
        crop_path = os.path.join(crops_dir, crop_filename)
        
        # Crop and save
        with Image.open(image_path) as img:
            bbox = (
                detection["xmin"], 
                detection["ymin"], 
                detection["xmax"], 
                detection["ymax"]
            )
            cropped = img.crop(bbox)
            cropped.save(crop_path)
        
        return crop_path