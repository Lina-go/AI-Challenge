"""
Modular crawler framework for modern slavery statements.

This package provides a flexible, extensible framework for crawling
modern slavery statements from different government repositories.

Key components:
- config: Configuration management for different jurisdictions
- base_crawler: Abstract base classes for crawler implementation
- adapters: Site-specific implementations for different jurisdictions
- utils: Common utilities for HTTP, HTML parsing, and data handling
- main: CLI entry point for running crawlers

Example usage:
    from crawlers.adapters import CanadianAdapter
    
    crawler = CanadianAdapter()
    data = crawler.crawl_and_save("canadian_statements.csv")
"""

from .config import CrawlerConfig, get_config, list_available_jurisdictions
from .base_crawler import BaseCrawler, FunctionalCrawler
from .adapters import CanadianAdapter

__version__ = "1.0.0"
__all__ = [
    "CrawlerConfig",
    "get_config",
    "list_available_jurisdictions",
    "BaseCrawler",
    "FunctionalCrawler",
    "CanadianAdapter",
]
