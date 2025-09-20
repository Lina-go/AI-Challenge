from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LLMConfig:
    """Configuration for LLM providers - Extended for all providers"""
    provider: str = "openai"  # openai, anthropic, huggingface, ollama
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.1
    
    # HuggingFace specific configurations
    base_url: Optional[str] = None  # For HuggingFace Router
    hf_token: Optional[str] = None  # HuggingFace token
    
    # Ollama specific configurations
    ollama_base_url: str = "http://localhost:11434"  # Ollama server URL
    ollama_timeout: int = 120  # Timeout for Ollama requests

@dataclass
class ProcessingConfig:
    """Configuration for PDF processing"""
    output_dir: str = "results"
    dpi: int = 300
    max_workers: int = 4
    timeout: int = 60
    save_page_images: bool = False

@dataclass
class TaskConfig:
    """Configuration for specific tasks"""
    task_type: str = "document_processing"  # document_processing, signature_extraction
    extract_tables: bool = True
    extract_figures: bool = True
    extract_signatures: bool = False
    debug_mode: bool = False

@dataclass
class ProcessorConfig:
    """Main configuration for PDF processor"""
    llm: LLMConfig
    processing: ProcessingConfig
    task: TaskConfig
    source_config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create_preset(cls, preset: str, output_dir: str = "results", **kwargs):
        """Create predefined configurations for common use cases"""
        presets = {
            # Commercial APIs for document processing
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
            ),
            
            # Ollama models
            "ollama-moondream": LLMConfig(
                provider="ollama",
                model="moondream:1.8b",
                ollama_base_url="http://localhost:11434"
            ),
            "ollama-llava": LLMConfig(
                provider="ollama", 
                model="llava:7b",
                ollama_base_url="http://localhost:11434"
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
            processing=ProcessingConfig(output_dir=output_dir),
            task=TaskConfig()
        )
    
    @classmethod
    def create_signature_preset(cls, preset: str, output_dir: str = "signature_results", **kwargs):
        """Create preset specifically for signature extraction"""
        config = cls.create_preset(preset, output_dir, **kwargs)
        config.task.task_type = "signature_extraction"
        config.task.extract_signatures = True
        config.task.extract_tables = False
        config.task.extract_figures = False
        return config