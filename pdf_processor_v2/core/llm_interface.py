"""
Unified LLM interface for both document processing and signature extraction.
"""

import logging
import base64
import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict
from PIL import Image
import openai
import anthropic

logger = logging.getLogger(__name__)

class LLMInterface(ABC):
    """Abstract interface for LLM interactions."""
    
    @abstractmethod
    def process_image_with_prompt(self, image: Image.Image, prompt: str) -> str:
        """
        Process an image with a text prompt.
        
        Args:
            image: PIL Image to process
            prompt: Text prompt for the LLM
            
        Returns:
            LLM response as string
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of the LLM provider."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is properly configured and available."""
        pass


class UnifiedLLMInterface(LLMInterface):
    """Unified interface for all LLM providers"""
    
    def __init__(self, llm_config):
        self.config = llm_config
        self.client = self._setup_client()
    
    def _setup_client(self):
        """Initialize the appropriate LLM client"""
        if self.config.provider.lower() == "openai":
            
            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            return openai.OpenAI(api_key=api_key)
            
        elif self.config.provider.lower() == "anthropic":
            
            api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key required")
            return anthropic.Anthropic(api_key=api_key)
            
        elif self.config.provider.lower() == "huggingface":
            api_key = self.config.hf_token or self.config.api_key or os.environ.get("HF_TOKEN")
            if not api_key:
                raise ValueError("HuggingFace token required. Set HF_TOKEN environment variable or provide hf_token in config")
            
            base_url = self.config.base_url or "https://router.huggingface.co/v1"
            return openai.OpenAI(
                base_url=base_url,
                api_key=api_key
            )
            
        elif self.config.provider.lower() == "ollama":
            try:
                import ollama
                return ollama
            except ImportError:
                raise ImportError("ollama package not installed. Install with: pip install ollama")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    def process_image_with_prompt(self, image: Image.Image, prompt: str) -> str:
        """Process image with text prompt using LLM vision capabilities"""
        if self.config.provider.lower() in ["openai", "huggingface"]:
            return self._process_with_openai_compatible(image, prompt)
        elif self.config.provider.lower() == "anthropic":
            return self._process_with_anthropic(image, prompt)
        elif self.config.provider.lower() == "ollama":
            return self._process_with_ollama(image, prompt)
    
    def _process_with_openai_compatible(self, image: Image.Image, prompt: str) -> str:
        """Process with OpenAI or OpenAI-compatible APIs (like HuggingFace Router)"""
        # Convert image to base64
        buffered = BytesIO()
        # Convert to RGB if necessary (removes alpha channel)
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error processing with {self.config.provider} ({self.config.model}): {e}")
            raise
    
    def _process_with_anthropic(self, image: Image.Image, prompt: str) -> str:
        """Process with Anthropic Claude Vision"""
        # Convert image to base64
        buffered = BytesIO()
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode()
        
        try:
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
                                    "media_type": "image/jpeg",
                                    "data": img_b64
                                }
                            },
                            {
                                "type": "text", 
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Error processing with Anthropic ({self.config.model}): {e}")
            raise
    
    def _process_with_ollama(self, image: Image.Image, prompt: str) -> str:
        """Process with Ollama local models"""
        import tempfile
        import os
        
        try:
            # Save image to temporary file (Ollama requires file path)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                if image.mode in ('RGBA', 'LA'):
                    image = image.convert('RGB')
                image.save(tmp_file.name, format="PNG")
                temp_image_path = tmp_file.name
            
            try:
                # Call Ollama chat API
                response = self.client.chat(
                    model=self.config.model,
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [temp_image_path]
                        }
                    ],
                    options={
                        'temperature': self.config.temperature,
                        'num_predict': self.config.max_tokens
                    }
                )
                
                return response['message']['content']
                
            finally:
                try:
                    os.unlink(temp_image_path)
                except OSError:
                    pass
                    
        except Exception as e:
            logger.error(f"Error processing with Ollama ({self.config.model}): {e}")
            raise
    
    def get_provider_name(self) -> str:
        """Get the name of the LLM provider."""
        return f"{self.config.provider.title()} ({self.config.model})"
    
    def is_available(self) -> bool:
        """Check if the LLM interface is properly configured and available."""
        try:
            if self.config.provider.lower() == "openai":
                api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
                return bool(api_key and api_key != 'your_openai_api_key_here')
            elif self.config.provider.lower() == "anthropic":
                api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
                return bool(api_key and api_key != 'your_anthropic_api_key_here')
            elif self.config.provider.lower() == "huggingface":
                api_key = self.config.hf_token or self.config.api_key or os.environ.get("HF_TOKEN")
                return bool(api_key)
            elif self.config.provider.lower() == "ollama":
                return self.client is not None
            return False
        except Exception:
            return False


def create_llm_interface(llm_config):
    """Factory function to create unified LLM interface"""
    return UnifiedLLMInterface(llm_config)