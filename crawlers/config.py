"""
Abstract configuration classes for the modular crawler framework.
"""

from dataclasses import dataclass
from typing import Optional

from .configs.canadian import create_config as create_canadian_config
from .configs.australian import create_config as create_australian_config

@dataclass
class CrawlerConfig:
    """
    Configuration class for site-specific crawler parameters.

    Attributes
    ----------
    name: str
        The name of the jurisdiction. E.g. "Canada Public Safety".
    jurisdiction_code: str
        The code of the jurisdiction. E.g. "CA" for Canada.
    search_url_template: str
        Template with {page} placeholder for the search URL.
    base_url: str
        The base URL for constructing full links.
    page_range: tuple
        The range of pages to crawl.
    results_per_page: int
        The number of results per page.
        By default, 50 results per page.
    delay_between_requests: float
        The delay between requests.
        By default, 1 second.
    request_timeout: int
        The timeout for requests.
        By default, 30 seconds.
    max_retries: int
        The maximum number of retries.
        By default, 3 retries.
    custom_headers: Optional[dict]
        Optional custom headers to add to requests.
    """

    name: str
    jurisdiction_code: str

    search_url_template: str
    base_url: str

    page_range: tuple
    results_per_page: int = 50

    delay_between_requests: float = 1.0
    request_timeout: int = 30
    max_retries: int = 3

    custom_headers: Optional[dict] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name or not self.jurisdiction_code:
            raise ValueError("Name and jurisdiction_code are required")

        if not self.search_url_template:
            raise ValueError("search_url_template is required")
        
        # Only require {page} placeholder if page_range spans multiple pages
        start_page, end_page = self.page_range
        if end_page > start_page and "{page}" not in self.search_url_template:
            raise ValueError("search_url_template must contain {page} placeholder when page_range spans multiple pages")

        if not self.base_url:
            raise ValueError("base_url is required")

        if start_page > end_page or start_page < 0:
            raise ValueError("Invalid page_range")

    @property
    def total_pages(self) -> int:
        """Calculate total number of pages to crawl."""
        start_page, end_page = self.page_range
        return end_page - start_page + 1


JURISDICTION_CONFIGS = {
    "canadian": create_canadian_config,
    "australian": create_australian_config,
    # 'british': create_british_config,
}
AVAILABLE_JURISDICTIONS = list(JURISDICTION_CONFIGS.keys())


def get_config(jurisdiction: str) -> CrawlerConfig:
    """
    Get configuration for a specific jurisdiction.

    Parameters
    ----------
    jurisdiction: str
        The name of the jurisdiction.

    Returns
    -------
    CrawlerConfig
        The configuration for the jurisdiction.

    Raises
    ------
    ValueError
        If jurisdiction is not supported.
    """
    jurisdiction_lower = jurisdiction.lower()

    if jurisdiction_lower not in JURISDICTION_CONFIGS:
        available = ", ".join(AVAILABLE_JURISDICTIONS)
        raise ValueError(
            f"Unsupported jurisdiction: {jurisdiction}. Available: {available}"
        )

    return JURISDICTION_CONFIGS[jurisdiction_lower]()


def list_available_jurisdictions() -> list:
    """
    List available jurisdictions.

    Returns
    -------
    list
        The list of available jurisdictions.
    """
    return AVAILABLE_JURISDICTIONS.copy()
