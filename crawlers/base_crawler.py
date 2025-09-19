"""
Refactored base crawler using modular components that follow SRP.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable

from bs4 import BeautifulSoup

from .config import CrawlerConfig
from .core import CrawlOrchestrator
from .utils import setup_crawler_logging

logger = logging.getLogger(__name__)


class BaseCrawler(ABC):
    """
    Abstract base class for web crawlers.

    Interface for site-specific extraction methods, provides a simple
    facade over the modular crawling components.
    """

    def __init__(self, config: CrawlerConfig):
        """
        Initialize crawler with configuration.

        Parameters
        ----------
        config : CrawlerConfig
            CrawlerConfig instance with site-specific settings.
        """
        self.config = config
        self._setup_logging()

        self.orchestrator = CrawlOrchestrator(
            config=config,
            link_extractor=self.extract_statement_links,
            data_extractor=self.extract_statement_data,
        )

    def _setup_logging(self) -> None:
        """Setup logging for this crawler instance."""
        setup_crawler_logging()

    @abstractmethod
    def extract_statement_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract statement page links from a search results page.

        Args:
            soup: BeautifulSoup object of the search page

        Returns:
            List of relative URLs to statement pages
        """
        pass

    @abstractmethod
    def extract_statement_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract data from an individual statement page.

        Args:
            soup: BeautifulSoup object of the statement page

        Returns:
            Dictionary of extracted data
        """
        pass

    # Delegate core functionality to orchestrator
    def crawl(self) -> List[Dict[str, Any]]:
        """
        Execute complete crawling process.

        Returns:
            List of extracted statement data
        """
        return self.orchestrator.execute_full_crawl()

    def _collect_and_save_links(self, output_dir: str) -> List[str]:
        """
        Collect statement URLs from search pages and save them to log file.
        
        Args:
            output_dir: Directory to save links file
            
        Returns:
            List of collected statement URLs
        """
        statement_urls = self.orchestrator.crawl_all_search_pages()
        
        if statement_urls:
            self.orchestrator.save_links_to_file(output_dir, statement_urls)
        
        return statement_urls

    def crawl_and_save(
        self,
        output_filename: str,
        file_format: str = "csv",
    ) -> List[Dict[str, Any]]:
        """
        Execute complete crawl and save results to file.
        Streaming is always enabled.

        Args:
            output_filename: Output file name
            file_format: Output format ("csv" or "json")

        Returns:
            List of extracted statement data
        """
        import os
        output_dir = os.path.dirname(output_filename) or "."
        
        # Always enable streaming
        log_file = self.orchestrator.enable_streaming_log(output_dir)
        print(f"📝 Streaming results to: {log_file}")

        # Collect and save statement URLs
        statement_urls = self._collect_and_save_links(output_dir)

        if statement_urls:
            # Process the statements
            data = self.orchestrator.crawl_all_statement_pages(statement_urls)

            # Save final results
            if data:
                from .utils import save_to_csv, save_to_json
                file_format = (
                    output_filename.split(".")[-1] if "." in output_filename else "csv"
                )
                
                if file_format.lower() == "json":
                    save_to_json(data, output_filename)
                else:
                    save_to_csv(data, output_filename)

            return data
        else:
            print("❌ No statement URLs found")
            return []


class FunctionalCrawler(BaseCrawler):
    """
    Crawler implementation that uses functional callbacks for extraction logic.

    This allows for easy adaptation to different sites without subclassing.
    """

    def __init__(
        self,
        config: CrawlerConfig,
        link_extractor: Callable[[BeautifulSoup], List[str]],
        data_extractor: Callable[[BeautifulSoup], Dict[str, Any]],
    ):
        """
        Initialize functional crawler.

        Args:
            config: CrawlerConfig instance
            link_extractor: Function to extract links from search pages
            data_extractor: Function to extract data from statement pages
        """
        self._link_extractor = link_extractor
        self._data_extractor = data_extractor

        super().__init__(config)

    def extract_statement_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract statement links using the provided function."""
        return self._link_extractor(soup)

    def extract_statement_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract statement data using the provided function."""
        return self._data_extractor(soup)
