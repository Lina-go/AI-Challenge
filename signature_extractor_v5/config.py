# signature_extractor_v5/config.py
"""Configuration management for signature extraction."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class DoclingConfig:
    """Docling VLM configuration."""
    model_name: str = "ibm-granite/granite-docling-258M"
    pipeline_type: str = "vlm"
    device: str = "auto"
    max_new_tokens: int = 4096
    temperature: float = 0.1
    use_cache: bool = True
    batch_size: int = 1


@dataclass
class ProcessingConfig:
    """Document processing configuration."""
    output_dir: str = "signature_results"
    max_pages: Optional[int] = None
    save_intermediate: bool = True
    timeout: int = 60


@dataclass
class ExtractionConfig:
    """Main extraction configuration."""
    docling: DoclingConfig = field(default_factory=DoclingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    def __post_init__(self):
        """Ensure output directory exists."""
        Path(self.processing.output_dir).mkdir(parents=True, exist_ok=True)