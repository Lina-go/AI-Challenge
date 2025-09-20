from pathlib import Path
from typing import Dict, List

def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory"""
    # Get prompts directory relative to this file
    prompts_dir = Path(__file__).parent.parent / "prompts"
    prompt_file = prompts_dir / f"{prompt_name}.txt"
    
    if not prompt_file.exists():
        available = get_available_prompts()
        raise FileNotFoundError(f"Prompt '{prompt_name}' not found. Available: {available}")
    
    return prompt_file.read_text(encoding='utf-8').strip()

def get_available_prompts() -> List[str]:
    """Get list of available prompt templates"""
    prompts_dir = Path(__file__).parent.parent / "prompts"
    
    if not prompts_dir.exists():
        return []
    
    return sorted([f.stem for f in prompts_dir.glob("*.txt")])

def load_all_prompts() -> Dict[str, str]:
    """Load all available prompts into a dictionary"""
    prompts = {}
    
    for prompt_name in get_available_prompts():
        prompts[prompt_name] = load_prompt(prompt_name)
    
    return prompts

def validate_prompt(prompt_content: str) -> bool:
    """Basic validation of prompt content"""
    return bool(prompt_content and prompt_content.strip() and len(prompt_content.strip()) >= 10)