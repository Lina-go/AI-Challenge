"""
Page crawling utilities for web crawling.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import requests
from bs4 import BeautifulSoup

from ..config import CrawlerConfig
from ..utils import make_request_with_retry, parse_html, make_absolute_urls

logger = logging.getLogger(__name__)


class PageCrawler:
    """
    Responsible for crawling individual web pages and extracting content.

    Single Responsibility: HTTP requests and HTML content extraction.
    """

    def __init__(self, session: requests.Session, config: CrawlerConfig):
        """
        Initialize page crawler.

        Parameters
        ----------
        session : requests.Session
            Configured requests session.
        config : CrawlerConfig
            CrawlerConfig instance with request settings.
        """
        self.session = session
        self.config = config

    def crawl_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Crawl a single page and return parsed HTML.

        Parameters
        ----------
        url : str
            URL to crawl.

        Returns
        -------
        BeautifulSoup or None
            BeautifulSoup object if successful, None if failed.
        """
        response = make_request_with_retry(
            session=self.session,
            url=url,
            timeout=self.config.request_timeout,
            max_retries=self.config.max_retries,
            delay=self.config.delay_between_requests,
        )

        if not response:
            return None

        return parse_html(response.content)

    def extract_links_from_search_page(
        self,
        search_url: str,
        link_extractor: Callable[[BeautifulSoup], List[str]],
        base_url: str,
    ) -> List[str]:
        """
        Crawl a search page and extract statement links.

        Parameters
        ----------
        search_url : str
            URL of the search page.
        link_extractor : Callable[[BeautifulSoup], List[str]]
            Function to extract relative URLs from soup.
        base_url : str
            Base URL for making URLs absolute.

        Returns
        -------
        List[str]
            List of absolute URLs to statement pages.
        """
        soup = self.crawl_page(search_url)
        if not soup:
            return []

        try:
            relative_urls = link_extractor(soup)
            absolute_urls = make_absolute_urls(relative_urls, base_url)

            logger.debug(f"Extracted {len(absolute_urls)} URLs from {search_url}")
            return absolute_urls

        except Exception as e:
            logger.error(f"Link extraction failed for {search_url}: {e}")
            return []

    def extract_data_from_statement_page(
        self,
        statement_url: str,
        data_extractor: Callable[[BeautifulSoup], Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Crawl a statement page and extract data.

        Parameters
        ----------
        statement_url : str
            URL of the statement page.
        data_extractor : Callable[[BeautifulSoup], Dict[str, Any]]
            Function to extract data from soup.

        Returns
        -------
        Dict[str, Any] or None
            Dictionary of extracted data if successful, None if failed.
        """
        soup = self.crawl_page(statement_url)
        if not soup:
            return None

        try:
            return data_extractor(soup)
        except Exception as e:
            logger.error(f"Data extraction failed for {statement_url}: {e}")
            return None
