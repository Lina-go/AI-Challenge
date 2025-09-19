"""
HTML parsing utilities for web crawling.
"""

import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def parse_html(content: bytes, parser: str = "html.parser") -> BeautifulSoup:
    """
    Parse HTML content into BeautifulSoup object.

    Parameters
    ----------
    content : bytes
        Raw HTML content as bytes.
    parser : str, default 'html.parser'
        Parser to use ('html.parser', 'lxml', etc.).

    Returns
    -------
    BeautifulSoup
        Parsed BeautifulSoup object.
    """
    return BeautifulSoup(content, parser)
