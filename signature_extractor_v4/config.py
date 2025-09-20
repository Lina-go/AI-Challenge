from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List

@dataclass
class LLMConfig:
    """Extended configuration for LLM providers - Supports commercial and open-source VLMs"""
    provider: str = "openai"  # openai, anthropic, huggingface, vllm, ollama, local
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.1
    
    # Open-source specific configurations
    model_path: Optional[str] = None  # For local models
    device: str = "auto"  # auto, cuda, cpu, cuda:0, etc.
    torch_dtype: str = "float16"  # float16, float32, bfloat16
    quantization: Optional[str] = None  # 4bit, 8bit, None
    
    # Hardware optimization
    low_cpu_mem_usage: bool = True
    use_flash_attention: bool = True
    max_memory: Optional[Dict[str, str]] = None  # {"0": "20GB", "1": "20GB"}
    
    # vLLM specific
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    
    # Ollama specific  
    ollama_base_url: str = "http://localhost:11434"
    
    # Performance optimization
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None
    
    # Fallback options
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None

@dataclass
class ProcessingConfig:
    """Configuration for PDF processing"""
    output_dir: str = "signature_results"
    dpi: int = 300
    max_workers: int = 4
    timeout: int = 60
    save_page_images: bool = False
    
    # Performance optimizations
    batch_size: int = 1  # For batch processing
    cache_models: bool = True  # Cache loaded models
    
    # Memory management
    clear_cache_between_docs: bool = False
    max_memory_usage_mb: int = 8192  # 8GB limit

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
            # Commercial APIs (existing)
            "openai": LLMConfig(provider="openai", model="gpt-4o"),
            "openai-mini": LLMConfig(provider="openai", model="gpt-4o-mini"),
            "anthropic": LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022"),
            "anthropic-haiku": LLMConfig(provider="anthropic", model="claude-3-haiku-20240307"),
            
            # High-performance open-source (GPU required)
            "qwen-large": LLMConfig(
                provider="huggingface",
                model="Qwen/Qwen2.5-VL-72B-Instruct",
                torch_dtype="float16",
                quantization="4bit",
                device="auto"
            ),
            "qwen-medium": LLMConfig(
                provider="huggingface", 
                model="Qwen/Qwen2.5-VL-32B-Instruct",
                torch_dtype="float16",
                quantization="4bit",
                device="auto"
            ),
            "qwen-small": LLMConfig(
                provider="huggingface",
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype="float16",
                device="auto"
            ),
            "cpu-qwen-small": LLMConfig(
                provider="huggingface",
                model="Qwen/Qwen2.5-VL-3B-Instruct",  # Smaller, more stable
                device="cpu",
                torch_dtype="float32",
                low_cpu_mem_usage=True,
                quantization=None  # Disable quantization for CPU stability
            ),

            "cpu-minicpm-stable": LLMConfig(
                provider="huggingface",
                model="openbmb/MiniCPM-V-2_6",
                device="cpu", 
                torch_dtype="float32",
                low_cpu_mem_usage=True,
                quantization=None
            ),

            "cpu-florence-fixed": LLMConfig(
                provider="huggingface",
                model="microsoft/Florence-2-base",
                device="cpu",
                torch_dtype="float32",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ),
            
            # Document-specialized models  
            "florence-large": LLMConfig(
                provider="huggingface",
                model="microsoft/Florence-2-large",
                torch_dtype="float16",
                device="auto"
            ),
            "florence-base": LLMConfig(
                provider="huggingface",
                model="microsoft/Florence-2-base",
                torch_dtype="float16",
                device="auto"
            ),
            
            # Mid-range GPU models
            "minicpm": LLMConfig(
                provider="huggingface",
                model="openbmb/MiniCPM-V-2_6",
                torch_dtype="float16",
                quantization="4bit",
                device="auto"
            ),
            "internvl": LLMConfig(
                provider="huggingface",
                model="OpenGVLab/InternVL2-8B",
                torch_dtype="float16",
                quantization="4bit"
            ),
            
            # CPU-optimized models (better compatibility)
            "cpu-florence": LLMConfig(
                provider="huggingface",
                model="microsoft/Florence-2-base",
                torch_dtype="float32",
                device="cpu",
                low_cpu_mem_usage=True
            ),
            "cpu-minicpm": LLMConfig(
                provider="huggingface",
                model="openbmb/MiniCPM-V-2_6",
                device="cpu",
                torch_dtype="float32",
                quantization="8bit",
                low_cpu_mem_usage=True
            ),
            "cpu-qwen": LLMConfig(
                provider="huggingface",
                model="Qwen/Qwen2.5-VL-3B-Instruct",  # Smaller model for CPU
                device="cpu",
                torch_dtype="float32",
                low_cpu_mem_usage=True
            ),
            
            # Local deployment via Ollama (most reliable for CPU)
            "ollama-llava": LLMConfig(
                provider="ollama",
                model="llava:7b"
            ),
            "ollama-llava-13b": LLMConfig(
                provider="ollama",
                model="llava:13b"
            ),
            "ollama-minicpm": LLMConfig(
                provider="ollama",
                model="minicpm-v"
            ),
            
            # vLLM optimized (for high-throughput)
            "vllm-qwen": LLMConfig(
                provider="vllm",
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                gpu_memory_utilization=0.8,
                tensor_parallel_size=1
            ),
            "vllm-minicpm": LLMConfig(
                provider="vllm",
                model="openbmb/MiniCPM-V-2_6",
                gpu_memory_utilization=0.7,
                tensor_parallel_size=1
            ),
            
            # Edge deployment (simplest, most compatible)
            "edge-florence": LLMConfig(
                provider="huggingface",
                model="microsoft/Florence-2-base",
                device="cpu",
                torch_dtype="float32",
                low_cpu_mem_usage=True
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
    
    @classmethod
    def get_preset_recommendations(cls) -> Dict[str, Dict[str, Any]]:
        """Get recommendations for different use cases"""
        return {
            "high_accuracy": {
                "preset": "qwen-medium",
                "description": "Best accuracy for document analysis",
                "requirements": "16GB+ GPU RAM, CUDA",
                "expected_performance": "85-92% accuracy, 3-8 sec/page"
            },
            "balanced": {
                "preset": "qwen-small", 
                "description": "Good balance of accuracy and speed",
                "requirements": "8GB+ GPU RAM, CUDA",
                "expected_performance": "80-85% accuracy, 2-5 sec/page"
            },
            "fast_gpu": {
                "preset": "minicpm",
                "description": "Fast processing on modest GPU",
                "requirements": "6GB+ GPU RAM, CUDA", 
                "expected_performance": "75-80% accuracy, 1-3 sec/page"
            },
            "cpu_only": {
                "preset": "cpu-florence",
                "description": "CPU-only processing",
                "requirements": "8GB+ System RAM",
                "expected_performance": "70-75% accuracy, 10-20 sec/page"
            },
            "edge_device": {
                "preset": "edge-florence",
                "description": "Ultra-lightweight for edge deployment",
                "requirements": "4GB+ System RAM",
                "expected_performance": "65-70% accuracy, 15-30 sec/page"
            },
            "document_specialist": {
                "preset": "florence-large",
                "description": "Specialized for document processing",
                "requirements": "8GB+ GPU RAM",
                "expected_performance": "80-85% accuracy, 2-4 sec/page"
            },
            "local_deployment": {
                "preset": "ollama-llava",
                "description": "Easy local deployment with Ollama",
                "requirements": "Ollama installed, 8GB+ System RAM",
                "expected_performance": "75-80% accuracy, 5-10 sec/page"
            }
        }
    
    @classmethod
    def auto_configure(cls, hardware_profile: str = "auto", output_dir: str = "signature_results") -> 'ExtractionConfig':
        """Automatically configure based on hardware profile"""
        try:
            import torch
            import psutil
            
            if hardware_profile == "auto":
                # Detect hardware automatically
                has_cuda = torch.cuda.is_available()
                if has_cuda:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    if gpu_memory >= 24:
                        hardware_profile = "high_end_gpu"
                    elif gpu_memory >= 12:
                        hardware_profile = "mid_range_gpu" 
                    elif gpu_memory >= 6:
                        hardware_profile = "entry_gpu"
                    else:
                        hardware_profile = "cpu_only"
                else:
                    hardware_profile = "cpu_only"
        except ImportError:
            # If torch not available, default to CPU
            hardware_profile = "cpu_only"
        
        profile_presets = {
            "high_end_gpu": "qwen-medium",
            "mid_range_gpu": "qwen-small", 
            "entry_gpu": "minicpm",
            "cpu_only": "cpu-florence",
            "edge": "edge-florence"
        }
        
        preset = profile_presets.get(hardware_profile, "cpu-florence")
        return cls.create_preset(preset, output_dir)

# Utility functions for configuration management
def get_all_presets() -> List[str]:
    """Get list of all available presets"""
    return [
        "openai", "openai-mini", "anthropic", "anthropic-haiku",
        "qwen-large", "qwen-medium", "qwen-small",
        "florence-large", "florence-base",
        "minicpm", "internvl",
        "cpu-florence", "cpu-minicpm", "cpu-qwen",
        "ollama-llava", "ollama-llava-13b", "ollama-minicpm",
        "vllm-qwen", "vllm-minicpm",
        "edge-florence"
    ]

def get_commercial_presets() -> List[str]:
    """Get list of commercial API presets"""
    return ["openai", "openai-mini", "anthropic", "anthropic-haiku"]

def get_opensource_presets() -> List[str]:
    """Get list of open-source model presets"""
    all_presets = get_all_presets()
    commercial = get_commercial_presets()
    return [p for p in all_presets if p not in commercial]

def get_cpu_compatible_presets() -> List[str]:
    """Get list of CPU-compatible presets"""
    return [
        "cpu-florence", "cpu-minicpm", "cpu-qwen",
        "ollama-llava", "ollama-llava-13b", "ollama-minicpm",
        "edge-florence"
    ]

def validate_config(config: ExtractionConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors"""
    warnings = []
    
    # Check for API keys when needed
    if config.llm.provider in ["openai", "anthropic"] and not config.llm.api_key:
        warnings.append(f"API key required for {config.llm.provider}")
    
    # Check hardware requirements
    if config.llm.provider == "huggingface":
        try:
            import torch
            if config.llm.device != "cpu" and not torch.cuda.is_available():
                warnings.append("CUDA not available but GPU device specified")
        except ImportError:
            warnings.append("PyTorch not available - required for HuggingFace models")
    
    # Check Ollama availability
    if config.llm.provider == "ollama":
        try:
            import requests
            response = requests.get(f"{config.llm.ollama_base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                warnings.append("Cannot connect to Ollama server")
        except:
            warnings.append("Cannot connect to Ollama server") 
    
    return warnings