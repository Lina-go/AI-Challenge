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
    
    def _get_model_auto_class(self, model_name: str):
        """Determine the appropriate auto class for the model"""
        model_name_lower = model_name.lower()
        
        # Vision-Language Models that need specific auto classes
        if any(keyword in model_name_lower for keyword in [
            'llava', 'llava-onevision', 'llava-next'
        ]):
            try:
                from transformers import LlavaForConditionalGeneration
                return LlavaForConditionalGeneration
            except ImportError:
                logger.warning("LlavaForConditionalGeneration not available, falling back to AutoModel")
                from transformers import AutoModel
                return AutoModel
        
        elif any(keyword in model_name_lower for keyword in [
            'idefics', 'smolvlm', 'granite-docling'
        ]):
            # These models use AutoModel for multimodal tasks
            from transformers import AutoModel
            return AutoModel
        
        elif any(keyword in model_name_lower for keyword in [
            'qwen2-vl', 'qwen2.5-vl'
        ]):
            try:
                from transformers import Qwen2VLForConditionalGeneration
                return Qwen2VLForConditionalGeneration
            except ImportError:
                logger.warning("Qwen2VLForConditionalGeneration not available, falling back to AutoModel")
                from transformers import AutoModel
                return AutoModel
        
        elif any(keyword in model_name_lower for keyword in [
            'florence'
        ]):
            try:
                from transformers import Florence2ForConditionalGeneration
                return Florence2ForConditionalGeneration
            except ImportError:
                logger.warning("Florence2ForConditionalGeneration not available, falling back to AutoModel")
                from transformers import AutoModel
                return AutoModel
        
        elif any(keyword in model_name_lower for keyword in [
            'minicpm-v', 'minicpm-llama3-v'
        ]):
            from transformers import AutoModel
            return AutoModel
        
        else:
            # Default for text-only or generic models
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM
    
    def _is_florence_model(self) -> bool:
        """Check if current model is Florence-2"""
        return 'florence' in self.config.model.lower()
    
    def _setup_huggingface(self):
        """Setup HuggingFace transformers model"""
        try:
            from transformers import AutoProcessor
            import torch
            
            # Device setup
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            # Torch dtype
            torch_dtype = getattr(torch, self.config.torch_dtype)
            
            # Get appropriate model class
            ModelClass = self._get_model_auto_class(self.config.model)
            
            # Model loading configuration
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
                "trust_remote_code": self.config.trust_remote_code,
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
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                except ImportError:
                    logger.warning("BitsAndBytesConfig not available, skipping 4bit quantization")
            elif self.config.quantization == "8bit":
                model_kwargs["load_in_8bit"] = True
            
            # Load processor first (works for most VLMs)
            logger.info(f"Loading processor for: {self.config.model}")
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.config.model, 
                    trust_remote_code=self.config.trust_remote_code
                )
            except Exception as e:
                logger.warning(f"Could not load AutoProcessor: {e}")
                # Some models might not have a processor
                self.processor = None
            
            # Load model with appropriate class
            logger.info(f"Loading HuggingFace model: {self.config.model} using {ModelClass.__name__}")
            
            try:
                if hasattr(ModelClass, 'from_pretrained'):
                    self.model = ModelClass.from_pretrained(
                        self.config.model,
                        **model_kwargs
                    )
                else:
                    # Fallback to AutoModel
                    from transformers import AutoModel
                    logger.warning(f"Falling back to AutoModel for {self.config.model}")
                    self.model = AutoModel.from_pretrained(
                        self.config.model,
                        **model_kwargs
                    )
                
                # Enable flash attention if available and requested
                if (self.config.use_flash_attention and 
                    hasattr(self.model, 'config') and 
                    hasattr(self.model.config, 'use_flash_attention_2')):
                    self.model.config.use_flash_attention_2 = True
                
                logger.info(f"Model loaded successfully on {device}")
                
            except Exception as model_error:
                logger.error(f"Failed to load model with {ModelClass.__name__}: {model_error}")
                
                # Final fallback: try AutoModel
                logger.info("Attempting fallback to AutoModel...")
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    self.config.model,
                    **model_kwargs
                )
                logger.info("Successfully loaded with AutoModel fallback")
            
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
                "trust_remote_code": True,
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
    
    def _convert_prompt_to_florence_task(self, prompt: str) -> str:
        """Convert general prompt to Florence-2 task format"""
        prompt_lower = prompt.lower()
        
        # Map common prompt patterns to Florence-2 tasks
        if any(keyword in prompt_lower for keyword in ['signature', 'sign', 'handwritten']):
            return "<OD>"  # Object detection for signatures
        elif any(keyword in prompt_lower for keyword in ['text', 'read', 'ocr']):
            return "<OCR>"  # OCR task
        elif any(keyword in prompt_lower for keyword in ['describe', 'caption', 'see']):
            return "<CAPTION>"  # General captioning
        elif any(keyword in prompt_lower for keyword in ['dense', 'detailed']):
            return "<DENSE_REGION_CAPTION>"  # Detailed description
        else:
            # Default to detailed captioning for document analysis
            return "<MORE_DETAILED_CAPTION>"
    
    def _process_with_huggingface(self, image: Image.Image, prompt: str) -> str:
        """Process with HuggingFace transformers model"""
        try:
            if not self.processor:
                raise Exception("No processor available for this model")
            
            # Special handling for Florence-2 models
            if self._is_florence_model():
                return self._process_with_florence(image, prompt)
            
            # Standard processing for other models
            try:
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                )
            except Exception as e:
                logger.warning(f"Standard processor call failed: {e}")
                # Try alternative processor call
                try:
                    inputs = self.processor(
                        prompt,
                        image,
                        return_tensors="pt"
                    )
                except Exception as e2:
                    logger.warning(f"Alternative processor call failed: {e2}")
                    # Final fallback - try with different parameter names
                    inputs = self.processor(
                        prompts=prompt,
                        images=image,
                        return_tensors="pt"
                    )
            
            # Move to device if CUDA
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
            
            # Generate response - handle different model interfaces
            with torch.no_grad():
                try:
                    # Standard generation
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        do_sample=self.config.temperature > 0,
                        pad_token_id=getattr(self.processor.tokenizer, 'eos_token_id', None)
                    )
                    
                    # Decode response
                    if 'input_ids' in inputs:
                        response = self.processor.decode(
                            outputs[0][inputs['input_ids'].shape[1]:], 
                            skip_special_tokens=True
                        )
                    else:
                        response = self.processor.decode(
                            outputs[0], 
                            skip_special_tokens=True
                        )
                    
                except Exception as gen_error:
                    logger.warning(f"Standard generation failed: {gen_error}")
                    
                    # Alternative: try with model.chat or model forward
                    try:
                        if hasattr(self.model, 'chat'):
                            response = self.model.chat(self.processor, image, prompt)
                        else:
                            # Direct forward pass
                            outputs = self.model(**inputs)
                            if hasattr(outputs, 'logits'):
                                # Simple greedy decoding
                                predicted_ids = torch.argmax(outputs.logits, dim=-1)
                                response = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
                            else:
                                response = str(outputs)
                    except Exception as alt_error:
                        logger.error(f"All generation methods failed: {alt_error}")
                        raise
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error processing with HuggingFace model: {e}")
            raise
    
    def _process_with_florence(self, image: Image.Image, prompt: str) -> str:
        """Special handling for Florence-2 models"""
        try:
            # Ensure RGB image
            if image.mode != "RGB":
                image = image.convert("RGB")

            # 1) Determine task token: prefer explicit token in prompt, otherwise infer
            explicit_tokens = [
                "<OCR>", "<OD>", "<CAPTION>", "<MORE_DETAILED_CAPTION>", "<DENSE_REGION_CAPTION>"
            ]
            task_prompt = None
            if prompt:
                for tok in explicit_tokens:
                    if tok in prompt:
                        task_prompt = tok
                        break
            if task_prompt is None:
                task_prompt = self._convert_prompt_to_florence_task(prompt or "")

            # 2) Florence expects ONLY the task token in the text field.
            # Do NOT append user text or <image> placeholder here.
            text = task_prompt

            # 3) Prepare inputs via processor
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
            )

            # 4) Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: (v.to(self.model.device) if hasattr(v, 'to') else v) for k, v in inputs.items()}

            # 5) Generate
            max_new = min(getattr(self.config, 'max_tokens', 512) or 512, 1024)
            do_sample = (getattr(self.config, 'temperature', 0.0) or 0.0) > 0
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    temperature=getattr(self.config, 'temperature', 0.0) or 0.0,
                    num_beams=1 if do_sample else 3,
                )

            # 6) Decode and post-process
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height),
            )

            # 7) Normalize outputs across tasks
            if task_prompt == "<OCR>":
                if isinstance(parsed, dict):
                    return parsed.get("text") or parsed.get("ocr") or ""
                return str(parsed)
            if task_prompt == "<OD>":
                # Return a succinct summary if possible
                if isinstance(parsed, dict):
                    labels = parsed.get("labels") or parsed.get("classes") or []
                    if labels:
                        return ", ".join(labels)
                return str(parsed)
            # Caption or other tasks
            if isinstance(parsed, dict):
                return parsed.get("text") or parsed.get("caption") or str(parsed)
            return str(parsed)

        except Exception as e:
            logger.error(f"Error processing with Florence-2: {e}")
            # Minimal fallback: force a plain caption with ONLY task token
            try:
                fallback_text = "<CAPTION>"
                inputs = self.processor(text=fallback_text, images=image, return_tensors="pt")
                if hasattr(self.model, 'device'):
                    inputs = {k: (v.to(self.model.device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                    )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return generated_text.strip()
            except Exception as fallback_error:
                logger.error(f"Florence-2 fallback also failed: {fallback_error}")
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
            "quantization": getattr(self.config, 'quantization', None),
            "model_class": type(self.model).__name__ if self.model else "Unknown"
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
        "microsoft/Florence-2-base",
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