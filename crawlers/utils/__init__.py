"""
Utilities package for the crawler framework.

This package contains focused utility modules, each handling a specific concern:
- http: HTTP request handling with retry logic
- html: HTML parsing utilities
- url: URL manipulation utilities
- data: Data processing and file I/O utilities
- logging: Logging configuration and progress tracking
"""

# Import commonly used functions for convenience
from .http import create_session, make_request_with_retry
from .html import parse_html
from .url import make_absolute_urls
from .data import clean_extracted_data, save_to_csv, save_to_json
from .logging import setup_crawler_logging, log_crawl_progress
from .streaming_logger import StreamingLogger, create_streaming_logger

__all__ = [
    # HTTP utilities
    "create_session",
    "make_request_with_retry",
    # HTML utilities
    "parse_html",
    # URL utilities
    "make_absolute_urls",
    # Data utilities
    "clean_extracted_data",
    "save_to_csv",
    "save_to_json",
    # Logging utilities
    "setup_crawler_logging",
    "log_crawl_progress",
    "StreamingLogger",
    "create_streaming_logger",
]
