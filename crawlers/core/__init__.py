"""
Core crawling components following Single Responsibility Principle.

This package contains focused classes, each with a single responsibility:

- URLGenerator: Generates URLs for crawling based on configuration
- PageCrawler: Handles HTTP requests and HTML parsing
- DataProcessor: Processes, cleans, and enriches extracted data
- CrawlOrchestrator: Coordinates the overall crawling workflow

This modular design makes the system more maintainable, testable, and flexible.
"""

from .url_generator import URLGenerator
from .page_crawler import PageCrawler
from .data_processor import DataProcessor
from .orchestrator import CrawlOrchestrator

__all__ = ["URLGenerator", "PageCrawler", "DataProcessor", "CrawlOrchestrator"]
