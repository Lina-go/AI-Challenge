import logging
import base64
import os
from io import BytesIO
from PIL import Image
import openai
import anthropic

logger = logging.getLogger(__name__)

class LLMInterface:
    """Simple interface for different LLM providers"""
    
    def __init__(self, llm_config):
        self.config = llm_config
        self.client = self._setup_client()
    
    def _setup_client(self):
        """Initialize the appropriate LLM client"""
        provider = self.config.provider.lower()
        
        if provider == "openai":
            return openai.OpenAI(api_key=self.config.api_key)
        
        elif provider == "anthropic":
            return anthropic.Anthropic(api_key=self.config.api_key)
        
        elif provider == "huggingface":
            api_key = self.config.hf_token or self.config.api_key or os.environ.get("HF_TOKEN")
            base_url = self.config.base_url or "https://router.huggingface.co/v1"
            return openai.OpenAI(base_url=base_url, api_key=api_key)
        
        elif provider == "ollama":
            import ollama
            return ollama
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def process_image_with_prompt(self, image: Image.Image, prompt: str) -> str:
        """Process image with text prompt using LLM vision capabilities"""
        provider = self.config.provider.lower()
        
        if provider in ["openai", "huggingface"]:
            return self._openai_compatible(image, prompt)
        elif provider == "anthropic":
            return self._anthropic(image, prompt)
        elif provider == "ollama":
            return self._ollama(image, prompt)
    
    def _openai_compatible(self, image: Image.Image, prompt: str) -> str:
        """OpenAI or HuggingFace Router"""
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]
            }],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        return response.choices[0].message.content
    
    def _anthropic(self, image: Image.Image, prompt: str) -> str:
        """Anthropic Claude Vision"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        message = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{
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
            }]
        )
        
        return message.content[0].text
    
    def _ollama(self, image: Image.Image, prompt: str) -> str:
        """Ollama local models"""
        import tempfile
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name, format="PNG")
            temp_path = tmp_file.name
        
        response = self.client.chat(
            model=self.config.model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [temp_path]
            }],
            options={
                'temperature': self.config.temperature,
                'num_predict': self.config.max_tokens
            }
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        return response['message']['content']

def create_llm_interface(llm_config):
    """Factory function to create LLM interface"""
    return LLMInterface(llm_config)

def get_available_providers():
    """Get list of available LLM providers"""
    return ["openai", "anthropic", "huggingface", "ollama"]