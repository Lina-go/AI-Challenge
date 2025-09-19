"""
Canadian modern slavery statements configuration.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import CrawlerConfig


def create_config() -> "CrawlerConfig":
    """
    Create configuration for Canadian modern slavery statements.

    Returns
    -------
    CrawlerConfig
        CrawlerConfig instance configured for Public Safety Canada's repository.
    """
    from ..config import CrawlerConfig
    return CrawlerConfig(
        name="Canada Public Safety",
        jurisdiction_code="CA",
        search_url_template="https://www.publicsafety.gc.ca/cnt/rsrcs/lbrr/ctlg/rslts-en.aspx?l=7&nb=50&pn={page}",
        base_url="https://www.publicsafety.gc.ca/cnt/rsrcs/lbrr/ctlg/",
        page_range=(1, 2),   # Sample: first 2 pages for schema demonstration
        results_per_page=50,
        delay_between_requests=0.2,
        request_timeout=30,
    )
