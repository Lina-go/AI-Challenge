"""
Logging utilities for web crawling.
"""

import logging
from typing import Optional


def setup_crawler_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration for crawler operations.

    Parameters
    ----------
    level : str, default 'INFO'
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    log_file : str, optional
        Optional log file path for file output.
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def log_crawl_progress(
    current: int, total: int, item_type: str = "items", interval: int = 100
) -> None:
    """
    Log crawling progress at regular intervals.

    Parameters
    ----------
    current : int
        Current item count.
    total : int
        Total item count.
    item_type : str, default 'items'
        Type of items being processed (e.g., "pages", "statements").
    interval : int, default 100
        Logging interval - log every N items.
    """
    logger = logging.getLogger(__name__)

    if current % interval == 0 or current == total:
        percentage = (current / total) * 100 if total > 0 else 0
        logger.info(f"Processed {current}/{total} {item_type} ({percentage:.1f}%)")
