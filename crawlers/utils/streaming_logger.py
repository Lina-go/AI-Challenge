"""
Simple streaming logger for incremental crawl results.
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime


class StreamingLogger:
    """
    Simple file logger that writes results immediately as they're found.

    Writes one line per successful statement extraction with just:
    - Timestamp
    - Title
    - URL
    """

    def __init__(self, log_file: str):
        """
        Initialize streaming logger.

        Parameters
        ----------
        log_file : str
            Path to log file (will be created/appended).
        """
        self.log_file = log_file
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Create log file with header if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("timestamp,title,url\n")

    def log_statement(self, data: Dict[str, Any]) -> None:
        """
        Log a successfully extracted statement immediately.

        Parameters
        ----------
        data : Dict[str, Any]
            Extracted statement data dictionary.
        """
        # Extract essential fields
        title = data.get("title") or data.get("statement_name") or "NO_TITLE"
        url = (
            data.get("pdf_url")
            or data.get("statement_url")
            or data.get("source_url")
            or "NO_URL"
        )

        # Clean up for CSV format (remove commas and newlines)
        title = (
            str(title).replace(",", ";").replace("\n", " ").replace("\r", " ").strip()
        )
        url = str(url).replace(",", ";").replace("\n", " ").replace("\r", " ").strip()

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write immediately to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{timestamp},{title},{url}\n")
            f.flush()  # Force write to disk immediately

    def get_count(self) -> int:
        """
        Get count of statements logged so far.

        Returns
        -------
        int
            Number of statements logged (excluding header).
        """
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return max(0, len(lines) - 1)  # Subtract header line
        except FileNotFoundError:
            return 0

    def get_latest_entries(self, count: int = 5) -> list:
        """
        Get the latest N entries from the log.

        Parameters
        ----------
        count : int, default 5
            Number of latest entries to return.

        Returns
        -------
        list
            List of latest entries as dictionaries.
        """
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) <= 1:  # Only header or empty
                return []

            # Get last N lines (excluding header)
            latest_lines = lines[-count:] if len(lines) > count else lines[1:]

            entries = []
            for line in latest_lines:
                parts = line.strip().split(",", 2)  # Split into max 3 parts
                if len(parts) >= 3:
                    entries.append(
                        {"timestamp": parts[0], "title": parts[1], "url": parts[2]}
                    )

            return entries

        except FileNotFoundError:
            return []


def create_streaming_logger(output_dir: str, jurisdiction: str) -> StreamingLogger:
    """
    Create a streaming logger with a timestamped filename.

    Parameters
    ----------
    output_dir : str
        Directory to create log file in.
    jurisdiction : str
        Jurisdiction code for filename.

    Returns
    -------
    StreamingLogger
        StreamingLogger instance.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{jurisdiction}_crawl_{timestamp}.log"
    log_path = os.path.join(output_dir, log_filename)

    return StreamingLogger(log_path)


def save_collected_links(output_dir: str, jurisdiction: str, links: list) -> str:
    """
    Save collected statement URLs to a log file.

    Parameters
    ----------
    output_dir : str
        Directory to create file in.
    jurisdiction : str
        Jurisdiction code for filename.
    links : list
        List of statement URLs.

    Returns
    -------
    str
        Path to the created links file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    links_filename = f"{jurisdiction}_statement_links_{timestamp}.log"
    links_path = os.path.join(output_dir, links_filename)

    with open(links_path, "w", encoding="utf-8") as f:
        f.write(f"# Statement URLs collected for {jurisdiction.upper()}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total URLs: {len(links)}\n")
        f.write("#\n")
        for link in links:
            f.write(f"{link}\n")

    return links_path
