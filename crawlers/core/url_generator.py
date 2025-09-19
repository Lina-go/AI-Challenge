"""
URL generation utilities for web crawling.
"""

import logging
from typing import List
from ..config import CrawlerConfig

logger = logging.getLogger(__name__)


class URLGenerator:
    """
    Responsible for generating URLs for crawling based on configuration.

    Single Responsibility: URL generation and template management.
    """

    def __init__(self, config: CrawlerConfig):
        """
        Initialize URL generator with configuration.

        Parameters
        ----------
        config : CrawlerConfig
            CrawlerConfig instance with URL templates and pagination info.
        """
        self.config = config

    def generate_search_urls(self) -> List[str]:
        """
        Generate list of search page URLs based on configuration.

        Returns
        -------
        List[str]
            List of search page URLs.
        """
        start_page, end_page = self.config.page_range
        search_urls = [
            self.config.search_url_template.format(page=page)
            for page in range(start_page, end_page + 1)
        ]

        logger.info(f"Generated {len(search_urls)} search URLs for {self.config.name}")
        return search_urls

    def get_base_url(self) -> str:
        """
        Get the base URL for constructing absolute URLs.

        Returns
        -------
        str
            Base URL string.
        """
        return self.config.base_url

    def get_page_count(self) -> int:
        """
        Calculate total number of pages to be crawled.

        Returns
        -------
        int
            Total page count.
        """
        start_page, end_page = self.config.page_range
        return end_page - start_page + 1
