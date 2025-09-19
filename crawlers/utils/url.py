"""
URL manipulation utilities for web crawling.
"""

from typing import List
from urllib.parse import urljoin


def make_absolute_urls(relative_urls: List[str], base_url: str) -> List[str]:
    """
    Convert relative URLs to absolute URLs using a base URL.

    Parameters
    ----------
    relative_urls : List[str]
        List of relative URLs to convert.
    base_url : str
        Base URL for joining with relative URLs.

    Returns
    -------
    List[str]
        List of absolute URLs.
    """
    return [urljoin(base_url, url) for url in relative_urls]
