"""
Australian modern slavery statements configuration.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import CrawlerConfig


def create_config(country: str = None) -> "CrawlerConfig":
    """
    Create configuration for Australian modern slavery statements.

    Parameters
    ----------
    country : str, optional
        Country to filter statements by (e.g., "United States of America").
        If None, downloads all statements.

    Returns
    -------
    CrawlerConfig
        CrawlerConfig instance configured for Australian Modern Slavery Register.
    """
    # Base parameters for the CSV download endpoint
    base_params = "q=&search_type=name&ordering=default&spsf=&spet=&voluntarity=&csv=1"
    
    # Add country filter if specified
    if country:
        country_param = f"&countries={country.replace(' ', '+')}"
        search_url = f"https://modernslaveryregister.gov.au/statements/?{base_params}{country_param}"
    else:
        search_url = f"https://modernslaveryregister.gov.au/statements/?{base_params}"
    
    # Australian system doesn't use pagination for CSV downloads - it's a single request
    from ..config import CrawlerConfig
    return CrawlerConfig(
        name="Australian Modern Slavery Register",
        jurisdiction_code="AU",
        search_url_template=search_url,  # No {page} placeholder needed for CSV download
        base_url="https://modernslaveryregister.gov.au/statements/",
        page_range=(1, 1),  # Single request only
        results_per_page=None,  # Not applicable for CSV download
        delay_between_requests=1.0,
        request_timeout=30,
        custom_headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1"
        }
    )
