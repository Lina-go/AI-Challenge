from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: str = "openai"  # openai, anthropic, etc.
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.1

@dataclass
class ProcessingConfig:
    """Configuration for PDF processing"""
    output_dir: str = "signature_results"
    dpi: int = 300
    max_workers: int = 4
    timeout: int = 60
    save_page_images: bool = False

@dataclass
class ExtractionConfig:
    """Main configuration for signature extraction"""
    llm: LLMConfig
    processing: ProcessingConfig
    source_config: Optional[Dict[str, Any]] = None