from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class YOLOModelConfig:
    """Configuration for different YOLO models."""
    model_name: str
    repo_id: str
    conf_threshold: float
    device_preference: str = "cpu"
    
    
class YOLOConfigRegistry:
    """Registry of available YOLO model configurations."""
    
    MODELS = {
        "signature_detector": YOLOModelConfig(
            model_name="signature_detector",
            repo_id="tech4humans/yolov8s-signature-detector",
            conf_threshold=0.6
        ),
        # Add more models as needed
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> YOLOModelConfig:
        """Get configuration for a specific model."""
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        return cls.MODELS[model_name]
    
    @classmethod
    def list_available_models(cls) -> list:
        """List all available model names."""
        return list(cls.MODELS.keys())
