import logging
import base64
import torch
from io import BytesIO
from typing import Any, Dict, Optional, Union
from PIL import Image
import openai
import anthropic

logger = logging.getLogger(__name__)

class LLMInterface:
    """Extended interface for commercial and open-source LLM providers"""
    
    def __init__(self, llm_config):
        self.config = llm_config
        self.client = None
        self.model = None
        self.processor = None
        self._setup_provider()
    
    def _setup_provider(self):
        """Initialize the appropriate LLM provider"""
        provider = self.config.provider.lower()
        
        if provider == "openai":
            self.client = openai.OpenAI(api_key=self.config.api_key)
        elif provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=self.config.api_key)
        elif provider == "huggingface":
            self._setup_huggingface()
        elif provider == "vllm":
            self._setup_vllm()
        elif provider == "ollama":
            self._setup_ollama()
        elif provider == "local":
            self._setup_local()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _setup_huggingface(self):
        """Setup HuggingFace transformers model"""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch
            
            # Device setup
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            # Torch dtype
            torch_dtype = getattr(torch, self.config.torch_dtype)
            
            # Model loading configuration
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            }
            
            # Device mapping for multi-GPU or CPU
            if device == "cpu":
                model_kwargs["device_map"] = "cpu"
            elif "cuda" in device:
                if self.config.max_memory:
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["max_memory"] = self.config.max_memory
                else:
                    model_kwargs["device_map"] = device
            
            # Quantization setup
            if self.config.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.config.quantization == "8bit":
                model_kwargs["load_in_8bit"] = True
            
            # Load model and processor
            logger.info(f"Loading HuggingFace model: {self.config.model}")
            self.processor = AutoProcessor.from_pretrained(self.config.model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                **model_kwargs
            )
            
            # Enable flash attention if available and requested
            if self.config.use_flash_attention and hasattr(self.model.config, 'use_flash_attention_2'):
                self.model.config.use_flash_attention_2 = True
            
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Error setting up HuggingFace model: {e}")
            raise
    
    def _setup_vllm(self):
        """Setup vLLM for optimized inference"""
        try:
            from vllm import LLM
            
            # vLLM configuration
            vllm_kwargs = {
                "model": self.config.model,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "dtype": self.config.torch_dtype,
            }
            
            logger.info(f"Loading vLLM model: {self.config.model}")
            self.model = LLM(**vllm_kwargs)
            logger.info("vLLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error setting up vLLM: {e}")
            raise
    
    def _setup_ollama(self):
        """Setup Ollama for local inference"""
        try:
            import requests
            
            # Test Ollama connection
            response = requests.get(f"{self.config.ollama_base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Cannot connect to Ollama server")
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            if self.config.model not in model_names:
                logger.warning(f"Model {self.config.model} not found in Ollama. Attempting to pull...")
                self._pull_ollama_model()
            
            self.client = self.config.ollama_base_url
            logger.info(f"Ollama setup complete for model: {self.config.model}")
            
        except Exception as e:
            logger.error(f"Error setting up Ollama: {e}")
            raise
    
    def _setup_local(self):
        """Setup for local model files"""
        # This could be extended for local ONNX, TensorRT, or other optimized formats
        logger.info("Local model setup - delegating to HuggingFace implementation")
        self._setup_huggingface()
    
    def _pull_ollama_model(self):
        """Pull model in Ollama"""
        import requests
        
        url = f"{self.config.ollama_base_url}/api/pull"
        payload = {"name": self.config.model}
        
        response = requests.post(url, json=payload, stream=True)
        if response.status_code == 200:
            logger.info(f"Successfully pulled model: {self.config.model}")
        else:
            raise Exception(f"Failed to pull model: {response.text}")
    
    def process_image_with_prompt(self, image: Image.Image, prompt: str) -> str:
        """Process image with text prompt using appropriate provider"""
        provider = self.config.provider.lower()
        
        if provider == "openai":
            return self._process_with_openai(image, prompt)
        elif provider == "anthropic":
            return self._process_with_anthropic(image, prompt)
        elif provider == "huggingface" or provider == "local":
            return self._process_with_huggingface(image, prompt)
        elif provider == "vllm":
            return self._process_with_vllm(image, prompt)
        elif provider == "ollama":
            return self._process_with_ollama(image, prompt)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _process_with_openai(self, image: Image.Image, prompt: str) -> str:
        """Process with OpenAI GPT-4 Vision"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_str}"}
                        }
                    ]
                }
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        return response.choices[0].message.content
    
    def _process_with_anthropic(self, image: Image.Image, prompt: str) -> str:
        """Process with Anthropic Claude Vision"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode()
        
        message = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        
        return message.content[0].text
    
    def _process_with_huggingface(self, image: Image.Image, prompt: str) -> str:
        """Process with HuggingFace transformers model"""
        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move to device if CUDA
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error processing with HuggingFace model: {e}")
            raise
    
    def _process_with_vllm(self, image: Image.Image, prompt: str) -> str:
        """Process with vLLM optimized inference"""
        try:
            from vllm import SamplingParams
            
            # Convert image to base64 for vLLM
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare prompt with image
            formatted_prompt = f"<image>{img_b64}</image>\n{prompt}"
            
            # Sampling parameters
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Generate
            outputs = self.model.generate([formatted_prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error processing with vLLM: {e}")
            raise
    
    def _process_with_ollama(self, image: Image.Image, prompt: str) -> str:
        """Process with Ollama"""
        try:
            import requests
            
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Ollama API call
            url = f"{self.config.ollama_base_url}/api/generate"
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            return response.json()["response"].strip()
            
        except Exception as e:
            logger.error(f"Error processing with Ollama: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            "provider": self.config.provider,
            "model": self.config.model,
            "device": getattr(self.config, 'device', 'unknown'),
            "quantization": getattr(self.config, 'quantization', None)
        }
        
        if hasattr(self.model, 'device'):
            info["actual_device"] = str(self.model.device)
        
        if hasattr(self.model, 'dtype'):
            info["dtype"] = str(self.model.dtype)
        
        return info

def create_llm_interface(llm_config):
    """Factory function to create LLM interface"""
    return LLMInterface(llm_config)

def get_available_providers():
    """Get list of available LLM providers"""
    return ["openai", "anthropic", "huggingface", "vllm", "ollama", "local"]

def get_cpu_optimized_models():
    """Get list of models optimized for CPU inference"""
    return [
        "HuggingFaceTB/SmolVLM-Instruct",
        "openbmb/MiniCPM-V-2_6", 
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "ibm-granite/granite-docling-258M"
    ]

def get_gpu_optimized_models():
    """Get list of models optimized for GPU inference"""
    return [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "microsoft/Florence-2-large",
        "OpenGVLab/InternVL2-8B"
    ]