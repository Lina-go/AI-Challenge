from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class DetectionConfig:
    """Configuration for signature detection"""
    model_name: str = "tech4humans/yolov8s-signature-detector"
    conf_threshold: float = 0.6
    ensure_one_per_page: bool = True
    device: str = "cpu"
    
    
@dataclass 
class ProcessingConfig:
    """Configuration for PDF processing"""
    output_dir: str = "signature_results"
    save_crops: bool = True
    save_page_images: bool = False
    max_workers: int = 4
    timeout: int = 30
    

@dataclass
class ExtractorConfig:
    """Main configuration for signature extraction"""
    detection: DetectionConfig
    processing: ProcessingConfig
    source_config: Optional[Dict[str, Any]] = None