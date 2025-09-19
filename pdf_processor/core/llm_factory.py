"""
Factory for creating LLM interfaces.
"""

import logging
from typing import Dict
from .llm_interface import LLMInterface

logger = logging.getLogger(__name__)


def create_llm_interface(provider: str = "gpt", **kwargs) -> LLMInterface:
    """
    Factory function to create LLM interfaces.
    
    Args:
        provider: LLM provider ("gpt", "gemma")
        **kwargs: Provider-specific arguments
        
    Returns:
        Configured LLM interface
        
    Raises:
        ValueError: If provider is not supported
        ImportError: If required dependencies are not installed
    """
    provider = provider.lower()
    
    if provider == "gpt":
        return _create_gpt_interface(**kwargs)
    elif provider == "gemma":
        return _create_gemma_interface(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported: gpt, gemma")


def _create_gpt_interface(**kwargs) -> LLMInterface:
    """Create GPT interface."""
    try:
        from .gpt_interface import create_gpt_interface
        return create_gpt_interface(**kwargs)
    except ImportError as e:
        logger.error(f"Failed to import GPT interface: {e}")
        raise ImportError("OpenAI library not installed. Run: pip install openai")


def _create_gemma_interface(**kwargs) -> LLMInterface:
    """Create Gemma interface."""
    try:
        from .gemma_interface import create_gemma_interface
        return create_gemma_interface(**kwargs)
    except ImportError as e:
        logger.error(f"Failed to import Gemma interface: {e}")
        raise ImportError("Transformers library not installed. Run: pip install torch transformers")


def get_available_providers() -> Dict[str, bool]:
    """
    Check which LLM providers are available.
    
    Returns:
        Dictionary mapping provider names to availability status
    """
    providers = {}
    
    # Check GPT
    try:
        gpt_interface = _create_gpt_interface()
        providers["gpt"] = gpt_interface.is_available()
    except Exception:
        providers["gpt"] = False
    
    # Check Gemma
    try:
        # Don't actually initialize Gemma (it's heavy), just check imports
        import transformers  # noqa: F401
        providers["gemma"] = True
    except Exception:
        providers["gemma"] = False
    
    return providers


def get_recommended_provider() -> str:
    """
    Get the recommended provider based on availability.
    
    Returns:
        Name of recommended provider
    """
    available = get_available_providers()
    
    # Prefer GPT if available (faster for testing)
    if available.get("gpt", False):
        return "gpt"
    elif available.get("gemma", False):
        return "gemma"
    else:
        logger.warning("No LLM providers are available")
        return "gpt"  # Default, will fail gracefully
