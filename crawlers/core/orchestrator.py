"""
Crawling orchestration utilities.
"""

import logging
from typing import List, Dict, Any, Callable, Optional

from bs4 import BeautifulSoup

from ..config import CrawlerConfig
from ..utils import create_session, log_crawl_progress, save_to_csv, save_to_json
from ..utils.streaming_logger import (
    StreamingLogger,
    create_streaming_logger,
    save_collected_links,
)
from .url_generator import URLGenerator
from .page_crawler import PageCrawler
from .data_processor import DataProcessor

logger = logging.getLogger(__name__)


class CrawlOrchestrator:
    """
    Responsible for orchestrating the complete crawling process.

    Single Responsibility: Workflow coordination and high-level crawling logic.
    """

    def __init__(
        self,
        config: CrawlerConfig,
        link_extractor: Callable[[BeautifulSoup], List[str]],
        data_extractor: Callable[[BeautifulSoup], Dict[str, Any]],
        streaming_logger: Optional[StreamingLogger] = None,
    ):
        """
        Initialize crawl orchestrator.

        Parameters
        ----------
        config : CrawlerConfig
            CrawlerConfig instance with crawling parameters.
        link_extractor : Callable[[BeautifulSoup], List[str]]
            Function to extract links from search pages.
        data_extractor : Callable[[BeautifulSoup], Dict[str, Any]]
            Function to extract data from statement pages.
        streaming_logger : StreamingLogger, optional
            Optional streaming logger for incremental results.
        """
        self.config = config
        self.link_extractor = link_extractor
        self.data_extractor = data_extractor
        self.streaming_logger = streaming_logger

        # Initialize components
        self.url_generator = URLGenerator(config)
        self.session = create_session(config.custom_headers)
        self.page_crawler = PageCrawler(self.session, config)
        self.data_processor = DataProcessor(config.jurisdiction_code)

    def crawl_all_search_pages(self) -> List[str]:
        """
        Crawl all search pages and collect statement URLs.

        Returns
        -------
        List[str]
            List of all statement page URLs.
        """
        logger.info(f"Starting search page crawl for {self.config.name}")

        search_urls = self.url_generator.generate_search_urls()
        all_statement_urls = []

        for i, search_url in enumerate(search_urls, 1):
            log_crawl_progress(i, len(search_urls), "search pages", interval=10)

            statement_urls = self.page_crawler.extract_links_from_search_page(
                search_url, self.link_extractor, self.url_generator.get_base_url()
            )
            all_statement_urls.extend(statement_urls)

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(all_statement_urls))

        logger.info(
            f"Found {len(unique_urls)} unique statement pages from {len(search_urls)} search pages"
        )
        return unique_urls

    def crawl_all_statement_pages(
        self, statement_urls: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Crawl all statement pages and extract data.

        Parameters
        ----------
        statement_urls : List[str]
            List of statement page URLs.

        Returns
        -------
        List[Dict[str, Any]]
            List of extracted and processed data dictionaries.
        """
        logger.info(f"Starting statement page crawl for {len(statement_urls)} pages")

        all_data = []
        successful_extractions = 0
        failed_extractions = 0

        for i, statement_url in enumerate(statement_urls, 1):
            log_crawl_progress(i, len(statement_urls), "statement pages", interval=100)

            # Extract raw data
            raw_data = self.page_crawler.extract_data_from_statement_page(
                statement_url, self.data_extractor
            )

            if raw_data:
                # Process and enrich the data
                processed_data = self.data_processor.process_statement_data(
                    raw_data, statement_url
                )

                if processed_data:
                    all_data.append(processed_data)
                    successful_extractions += 1

                    # Log immediately to streaming log if available
                    if self.streaming_logger:
                        self.streaming_logger.log_statement(processed_data)

                        # Show running count every 50 statements
                        if successful_extractions % 50 == 0:
                            count = self.streaming_logger.get_count()
                            print(f"✅ Logged {count} statements to file...")
                else:
                    failed_extractions += 1
            else:
                failed_extractions += 1

        # Calculate and log statistics
        stats = self.data_processor.calculate_statistics(
            successful_extractions, failed_extractions
        )
        logger.info(
            f"Crawl completed: {stats['successful_extractions']} successful, "
            f"{stats['failed_extractions']} failed ({stats['success_rate']:.1f}% success rate)"
        )

        return all_data

    def execute_full_crawl(self) -> List[Dict[str, Any]]:
        """
        Execute complete crawling process.

        Returns
        -------
        List[Dict[str, Any]]
            List of extracted statement data.
        """
        logger.info(f"Starting complete crawl for {self.config.name}")

        # Step 1: Crawl all search pages to get statement URLs
        statement_urls = self.crawl_all_search_pages()

        if not statement_urls:
            logger.warning("No statement URLs found")
            return []

        # Step 2: Crawl all statement pages to extract data
        all_data = self.crawl_all_statement_pages(statement_urls)

        logger.info(f"Crawl completed: {len(all_data)} statements extracted")
        return all_data

    def execute_and_save(
        self, output_filename: str, file_format: str = "csv"
    ) -> List[Dict[str, Any]]:
        """
        Execute complete crawl and save results to file.

        Args:
            output_filename: Output file name
            file_format: Output format ("csv" or "json")

        Returns:
            List of extracted statement data
        """
        data = self.execute_full_crawl()

        if data:
            if file_format.lower() == "csv":
                save_to_csv(data, output_filename)
            elif file_format.lower() == "json":
                save_to_json(data, output_filename)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        else:
            logger.warning("No data to save")

        return data

    def enable_streaming_log(self, output_dir: str) -> str:
        """
        Enable streaming log to write results immediately.

        Parameters
        ----------
        output_dir : str
            Directory to create log file in.

        Returns
        -------
        str
            Path to the created log file.
        """
        self.streaming_logger = create_streaming_logger(
            output_dir, self.config.jurisdiction_code
        )
        return self.streaming_logger.log_file

    def save_links_to_file(self, output_dir: str, links: list) -> str:
        """
        Save collected statement URLs to a file.

        Parameters
        ----------
        output_dir : str
            Directory to create file in.
        links : list
            List of statement URLs.

        Returns
        -------
        str
            Path to the created links file.
        """
        links_file = save_collected_links(
            output_dir, self.config.jurisdiction_code, links
        )
        print(f"📋 Saved {len(links)} collected URLs to: {links_file}")
        return links_file
