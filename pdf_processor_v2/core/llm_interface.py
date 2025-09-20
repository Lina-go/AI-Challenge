"""
LLM interface for processing page images with prompts.
"""

from abc import ABC, abstractmethod
from PIL import Image


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
