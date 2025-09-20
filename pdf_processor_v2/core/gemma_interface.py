"""
Gemma LLM interface for processing page images with prompts.
"""

import tempfile
import os
from transformers import pipeline
import torch
from PIL import Image
from .llm_interface import LLMInterface


class GemmaLLMInterface(LLMInterface):
    """
    Gemma LLM interface for processing images with text prompts.
    Uses google/gemma-3-4b-it model for vision-language tasks.
    """
    
    def __init__(self, model_name: str = "google/gemma-3-4b-it", device: str = "cpu"):
        """
        Initialize Gemma LLM interface.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self.pipe = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Gemma model pipeline."""
        print(f"Loading Gemma model: {self.model_name}")
        print("This may take a few minutes on first run...")
        
        self.pipe = pipeline(
            "image-text-to-text",
            model=self.model_name,
            device=self.device,
            torch_dtype=torch.bfloat16,
            use_fast=True
        )
        
        self.pipe.model.eval()
        print("Model loaded successfully!")
    
    def process_image_with_prompt(self, image: Image.Image, prompt: str) -> str:
        """
        Process an image with a text prompt using Gemma.
        
        Args:
            image: PIL Image to process
            prompt: Text prompt for the LLM
            
        Returns:
            LLM response as string
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name, 'PNG')
            img_path = tmp_file.name
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant specialized in analyzing document images and extracting structured information."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": img_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            with torch.no_grad():
                output = self.pipe(text=messages, max_new_tokens=2000)
            
            response = output[0]["generated_text"][-1]["content"]
            
            return response.strip()
            
        finally:
            try:
                os.unlink(img_path)
            except OSError:
                pass  # Ignore cleanup errors
    
    def get_provider_name(self) -> str:
        """Get the name of the LLM provider."""
        return f"Gemma ({self.model_name})"
    
    def is_available(self) -> bool:
        """Check if the Gemma interface is properly configured and available."""
        try:
            return self.pipe is not None
        except Exception:
            return False


def create_gemma_interface(device: str = "cpu") -> GemmaLLMInterface:
    """
    Create a Gemma LLM interface with automatic device detection.
    
    Args:
        device: Device preference ("cpu", "cuda", or "auto")
        
    Returns:
        Initialized GemmaLLMInterface
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {device}")
    
    return GemmaLLMInterface(device=device)