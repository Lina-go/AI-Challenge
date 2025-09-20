import os
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt template from the prompts directory
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        
    Returns:
        Prompt content as string
    """
    try:
        # Get the directory where this module is located
        module_dir = Path(__file__).parent.parent
        prompts_dir = module_dir / "prompts"
        
        prompt_file = prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            available_prompts = get_available_prompts()
            raise FileNotFoundError(
                f"Prompt '{prompt_name}' not found. Available prompts: {available_prompts}"
            )
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        logger.debug(f"Loaded prompt: {prompt_name}")
        return content
        
    except Exception as e:
        logger.error(f"Error loading prompt '{prompt_name}': {e}")
        raise

def get_available_prompts() -> List[str]:
    """
    Get list of available prompt templates
    
    Returns:
        List of prompt names (without .txt extension)
    """
    try:
        module_dir = Path(__file__).parent.parent
        prompts_dir = module_dir / "prompts"
        
        if not prompts_dir.exists():
            return []
        
        prompt_files = list(prompts_dir.glob("*.txt"))
        prompt_names = [f.stem for f in prompt_files]
        
        return sorted(prompt_names)
        
    except Exception as e:
        logger.error(f"Error getting available prompts: {e}")
        return []