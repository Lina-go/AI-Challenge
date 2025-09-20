from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LLMConfig:
    """Configuration for LLM providers - Extended for HuggingFace Router"""
    provider: str = "openai"  # openai, anthropic, huggingface
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.1
    
    # HuggingFace specific configurations
    base_url: Optional[str] = None  # For HuggingFace Router
    hf_token: Optional[str] = None  # HuggingFace token

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
    
    # Model presets for easy selection
    @classmethod
    def create_preset(cls, preset: str, output_dir: str = "signature_results", **kwargs):
        """Create predefined configurations for common use cases"""
        presets = {
            # Commercial APIs
            "openai": LLMConfig(provider="openai", model="gpt-4o"),
            "anthropic": LLMConfig(provider="anthropic", model="claude-3-sonnet"),
            
            # HuggingFace Router models
            "glm-4.5v": LLMConfig(
                provider="huggingface",
                model="zai-org/GLM-4.5V:novita",
                base_url="https://router.huggingface.co/v1"
            ),
            "qwen-vl": LLMConfig(
                provider="huggingface", 
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                base_url="https://router.huggingface.co/v1"
            )
        }
        
        if preset not in presets:
            available = list(presets.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
        
        # Override with any custom kwargs
        llm_config = presets[preset]
        for key, value in kwargs.items():
            if hasattr(llm_config, key):
                setattr(llm_config, key, value)
        
        return cls(
            llm=llm_config,
            processing=ProcessingConfig(output_dir=output_dir)
        )