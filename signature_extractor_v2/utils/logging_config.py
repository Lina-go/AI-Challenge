import logging
import logging.config
from pathlib import Path
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None
):
    """
    Setup logging configuration for the signature extractor
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file name
        log_dir: Optional log directory (defaults to logs/)
    """
    
    # Create logs directory if specified
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    elif log_file:
        log_dir = "logs"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = f"{log_dir}/{log_file}"
    
    # Define logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            'signature_extractor_v2': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': level,
            'handlers': ['console']
        }
    }
    
    # Add file handler if log file specified
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'detailed',
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
        config['loggers']['signature_extractor_v2']['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    logging.config.dictConfig(config)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"signature_extractor_v2.{name}")

def set_log_level(level: str):
    """
    Change the logging level for all signature_extractor loggers
    
    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logger = logging.getLogger('signature_extractor_v2')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(getattr(logging, level.upper()))