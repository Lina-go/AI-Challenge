import logging
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    '''Setup logging configuration for signature extraction.'''
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    '''Get a logger instance.'''
    return logging.getLogger(name)