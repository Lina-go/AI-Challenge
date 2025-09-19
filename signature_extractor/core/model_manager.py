import os
import glob
from ultralytics import YOLO
from huggingface_hub import snapshot_download


class ModelManager:
    """
    Manages YOLO model downloading and loading.
    
    Single Responsibility: Model lifecycle management.
    """
    
    def __init__(self, config: ExtractorConfig):
        self.config = config
        self.model = None
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Download model if not available locally."""
        local_dir = "yolov8s-signature-detector"
        
        if not os.path.exists(local_dir):
            logger.info(f"Downloading model: {self.config.detection.model_name}")
            snapshot_download(
                self.config.detection.model_name,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                etag_timeout=60
            )
        
        # Find the model weights file
        weights_path = glob.glob(os.path.join(local_dir, "**", "*.pt"), recursive=True)[0]
        self.model = YOLO(weights_path)
        logger.info(f"Model loaded from: {weights_path}")
    
    def get_model(self) -> YOLO:
        """Get the loaded YOLO model."""
        return self.model
