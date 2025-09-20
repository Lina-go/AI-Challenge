"""
Configuration modules for signature extraction.

This package provides configuration management for different aspects
of the signature extraction system:

- default_config: Default extraction configuration and factory functions
- llm_config: LLM provider configurations and model settings

Configuration follows a hierarchical approach:
1. Package defaults
2. Environment variables  
3. Configuration files
4. Command line arguments
5. Programmatic overrides
"""

from .default_config import (
    create_default_config,
    get_config_template,
    validate_config
)

from .llm_config import (
    LLMProviderConfig,
    LLMModelRegistry,
    get_provider_config,
    list_available_models
)

__all__ = [
    # Default configuration
    "create_default_config",
    "get_config_template", 
    "validate_config",
    
    # LLM configuration
    "LLMProviderConfig",
    "LLMModelRegistry",
    "get_provider_config",
    "list_available_models"
]

# Configuration constants
DEFAULT_OUTPUT_DIR = "signature_results"
DEFAULT_DPI = 300
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_WORKERS = 4

# Environment variable prefix
ENV_PREFIX = "SIGNATURE_EXTRACTOR_"

# Supported configuration formats
SUPPORTED_CONFIG_FORMATS = ["json", "yaml", "toml"]

def load_config_from_file(config_path: str, config_format: str = None):
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        config_format: Format of config file (json, yaml, toml) or auto-detect
        
    Returns:
        Configuration dictionary
    """
    from pathlib import Path
    import json
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Auto-detect format if not specified
    if config_format is None:
        config_format = path.suffix.lstrip('.').lower()
    
    if config_format not in SUPPORTED_CONFIG_FORMATS:
        raise ValueError(f"Unsupported config format: {config_format}")
    
    with open(path, 'r', encoding='utf-8') as f:
        if config_format == "json":
            return json.load(f)
        elif config_format == "yaml":
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML configuration files")
        elif config_format == "toml":
            try:
                import tomli
                return tomli.load(f)
            except ImportError:
                raise ImportError("tomli required for TOML configuration files")

def save_config_to_file(config_dict: dict, config_path: str, config_format: str = "json"):
    """
    Save configuration to file.
    
    Args:
        config_dict: Configuration dictionary to save
        config_path: Path where to save configuration
        config_format: Format to save in (json, yaml, toml)
    """
    from pathlib import Path
    import json
    
    if config_format not in SUPPORTED_CONFIG_FORMATS:
        raise ValueError(f"Unsupported config format: {config_format}")
    
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        if config_format == "json":
            json.dump(config_dict, f, indent=2, default=str)
        elif config_format == "yaml":
            try:
                import yaml
                yaml.dump(config_dict, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML required for YAML configuration files")
        elif config_format == "toml":
            try:
                import tomli_w
                tomli_w.dump(config_dict, f)
            except ImportError:
                raise ImportError("tomli-w required for TOML configuration files")

def load_config_from_env():
    """
    Load configuration from environment variables.
    
    Returns:
        Configuration dictionary with values from environment
    """
    import os
    
    env_config = {}
    
    # LLM configuration
    if f"{ENV_PREFIX}LLM_PROVIDER" in os.environ:
        env_config.setdefault("llm", {})["provider"] = os.environ[f"{ENV_PREFIX}LLM_PROVIDER"]
    
    if f"{ENV_PREFIX}LLM_MODEL" in os.environ:
        env_config.setdefault("llm", {})["model"] = os.environ[f"{ENV_PREFIX}LLM_MODEL"]
    
    if f"{ENV_PREFIX}API_KEY" in os.environ:
        env_config.setdefault("llm", {})["api_key"] = os.environ[f"{ENV_PREFIX}API_KEY"]
    
    # Processing configuration  
    if f"{ENV_PREFIX}OUTPUT_DIR" in os.environ:
        env_config.setdefault("processing", {})["output_dir"] = os.environ[f"{ENV_PREFIX}OUTPUT_DIR"]
    
    if f"{ENV_PREFIX}DPI" in os.environ:
        env_config.setdefault("processing", {})["dpi"] = int(os.environ[f"{ENV_PREFIX}DPI"])
    
    if f"{ENV_PREFIX}MAX_WORKERS" in os.environ:
        env_config.setdefault("processing", {})["max_workers"] = int(os.environ[f"{ENV_PREFIX}MAX_WORKERS"])
    
    return env_config

def get_effective_config(**overrides):
    """
    Get effective configuration by merging defaults, environment, and overrides.
    
    Args:
        **overrides: Configuration overrides
        
    Returns:
        Merged configuration dictionary
    """
    # Start with defaults
    config = get_config_template()
    
    # Apply environment variables
    env_config = load_config_from_env()
    _deep_update(config, env_config)
    
    # Apply overrides
    _deep_update(config, overrides)
    
    return config

def _deep_update(base_dict: dict, update_dict: dict):
    """Recursively update nested dictionary."""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value