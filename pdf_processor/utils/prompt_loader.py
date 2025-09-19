"""
Prompt loading and management utilities.
"""

from pathlib import Path
from typing import Dict


class PromptLoader:
    """Handles loading and caching of prompt templates."""
    
    def __init__(self, prompts_dir: str = "pdf_processor/prompts"):
        """
        Initialize prompt loader.
        
        Args:
            prompts_dir: Directory containing prompt files
        """
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt template from file.
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            
        Returns:
            Prompt content as string
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        if prompt_name in self._cache:
            return self._cache[prompt_name]
        
        prompt_path = self.prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            self._cache[prompt_name] = content
            return content
            
        except Exception as e:
            raise Exception(f"Failed to load prompt {prompt_name}: {e}")
    
    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._cache.clear()
    
    def get_available_prompts(self) -> list[str]:
        """Get list of available prompt files."""
        if not self.prompts_dir.exists():
            return []
        
        return [f.stem for f in self.prompts_dir.glob("*.txt")]


# Global prompt loader instance
_prompt_loader = PromptLoader()


def load_prompt(prompt_name: str) -> str:
    """
    Convenience function to load a prompt.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        
    Returns:
        Prompt content as string
    """
    return _prompt_loader.load_prompt(prompt_name)


def get_available_prompts() -> list[str]:
    """Get list of available prompt files."""
    return _prompt_loader.get_available_prompts()
