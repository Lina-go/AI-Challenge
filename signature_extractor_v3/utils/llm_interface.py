import logging
import base64
from io import BytesIO
from typing import Any, Dict
from PIL import Image
import openai
import anthropic

logger = logging.getLogger(__name__)

class LLMInterface:
    """Interface for different LLM providers"""
    
    def __init__(self, llm_config):
        self.config = llm_config
        self.client = self._setup_client()
    
    def _setup_client(self):
        """Initialize the appropriate LLM client"""
        if self.config.provider.lower() == "openai":
            return openai.OpenAI(api_key=self.config.api_key)
        elif self.config.provider.lower() == "anthropic":
            return anthropic.Anthropic(api_key=self.config.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    def process_image_with_prompt(self, image: Image.Image, prompt: str) -> str:
        """Process image with text prompt using LLM vision capabilities"""
        if self.config.provider.lower() == "openai":
            return self._process_with_openai(image, prompt)
        elif self.config.provider.lower() == "anthropic":
            return self._process_with_anthropic(image, prompt)
    
    def _process_with_openai(self, image: Image.Image, prompt: str) -> str:
        """Process with OpenAI GPT-4 Vision"""
        # Convert image to base64
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
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
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
        # Convert image to base64
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
                        {
                            "type": "text", 
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        return message.content[0].text

def create_llm_interface(llm_config):
    """Factory function to create LLM interface"""
    return LLMInterface(llm_config)

def get_available_providers():
    """Get list of available LLM providers"""
    return ["openai", "anthropic"]