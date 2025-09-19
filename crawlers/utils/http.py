"""
HTTP utilities for web crawling.
"""

import time
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def create_session(custom_headers: Optional[dict] = None) -> requests.Session:
    """
    Create a configured requests session.

    Parameters
    ----------
    custom_headers : dict, optional
        Optional custom headers to add to the session.

    Returns
    -------
    requests.Session
        Configured requests session with default and custom headers.
    """
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    if custom_headers:
        session.headers.update(custom_headers)

    return session


def make_request_with_retry(
    session: requests.Session,
    url: str,
    timeout: int = 30,
    max_retries: int = 3,
    delay: float = 1.0,
) -> Optional[requests.Response]:
    """
    Make HTTP request with retry logic and exponential backoff.

    Parameters
    ----------
    session : requests.Session
        Requests session to use for making the request.
    url : str
        URL to request.
    timeout : int, default 30
        Request timeout in seconds.
    max_retries : int, default 3
        Maximum number of retry attempts.
    delay : float, default 1.0
        Initial delay between requests in seconds.

    Returns
    -------
    requests.Response or None
        Response object if successful, None if all attempts failed.
    """
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                # Exponential backoff for retries
                time.sleep(delay * (2 ** (attempt - 1)))

            # Rate limiting delay
            time.sleep(delay)

            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response

        except requests.RequestException as e:
            if attempt == max_retries:
                logger.error(
                    f"Request failed after {max_retries + 1} attempts for {url}: {e}"
                )
                return None
            else:
                logger.warning(
                    f"Request attempt {attempt + 1} failed for {url}: {e}. Retrying..."
                )

    return None
