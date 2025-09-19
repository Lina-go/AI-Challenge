"""
OpenAI GPT interface for processing page images with prompts.
"""

import os
import base64
import io
import logging
from typing import Optional
from PIL import Image

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class GPTInterface(LLMInterface):
    """OpenAI GPT interface for vision tasks."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize GPT interface.
        
        Args:
            model: GPT model to use (default: gpt-4o for vision)
            api_key: OpenAI API key (if None, will try to load from environment)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.model = model
        
        # Set up API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self._load_api_key()
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized GPT interface with model: {self.model}")
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment or config file."""
        # Try environment variable first
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key != 'your_openai_api_key_here':
            return api_key
        
        # Try loading from config.env file
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.env')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            key = line.split('=', 1)[1].strip()
                            if key and key != 'your_openai_api_key_here':
                                return key
        except Exception as e:
            logger.debug(f"Could not load config file: {e}")
        
        return None
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        # Convert to RGB if necessary (removes alpha channel)
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def process_image_with_prompt(self, image: Image.Image, prompt: str) -> str:
        """
        Process an image with a text prompt using GPT-4 Vision.
        
        Args:
            image: PIL Image to process
            prompt: Text prompt for the LLM
            
        Returns:
            LLM response as string
        """
        try:
            # Convert image to base64
            image_b64 = self._image_to_base64(image)
            
            # Create the message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4000,
                temperature=0.1  # Low temperature for consistent results
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"GPT API call failed: {e}")
            return f"GPT processing failed: {e}"
    
    def get_provider_name(self) -> str:
        """Get the name of the LLM provider."""
        return f"OpenAI GPT ({self.model})"
    
    def is_available(self) -> bool:
        """Check if the GPT interface is properly configured and available."""
        if not OPENAI_AVAILABLE:
            return False
        
        if not self.api_key or self.api_key == 'your_openai_api_key_here':
            return False
        
        # Test API connection with a simple call
        try:
            test_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use cheaper model for test
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"GPT availability check failed: {e}")
            return False


def create_gpt_interface(model: str = "gpt-4o", api_key: Optional[str] = None) -> GPTInterface:
    """
    Factory function to create GPT interface.
    
    Args:
        model: GPT model to use
        api_key: Optional API key
        
    Returns:
        Configured GPT interface
    """
    return GPTInterface(model=model, api_key=api_key)
