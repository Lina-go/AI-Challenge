"""
Data processing utilities for web crawling.
"""

import json
import logging
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def clean_extracted_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and normalize extracted data by removing empty values and trimming strings.

    Parameters
    ----------
    data : Dict[str, Any]
        Raw extracted data dictionary.

    Returns
    -------
    Dict[str, Any]
        Cleaned data dictionary with empty values removed.
    """
    cleaned = {}

    for key, value in data.items():
        if value is not None:
            # Clean string values
            if isinstance(value, str):
                cleaned_value = value.strip()
                if cleaned_value:  # Only include non-empty strings
                    cleaned[key] = cleaned_value
            else:
                cleaned[key] = value

    return cleaned


def save_to_csv(
    data: List[Dict[str, Any]], filename: str, show_preview: bool = True
) -> None:
    """
    Save data to CSV file with optional preview display.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        List of data dictionaries to save.
    filename : str
        Output filename for the CSV file.
    show_preview : bool, default True
        Whether to show data preview after saving.

    Raises
    ------
    Exception
        If saving fails.
    """
    if not data:
        logger.warning("No data to save")
        return

    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(data)} records to {filename}")

        if show_preview:
            print(f"\nData preview for {filename}:")
            print(df.head())
            print(f"\nColumns: {list(df.columns)}")
            print(f"Total records: {len(df)}")

    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        raise


def save_to_json(data: List[Dict[str, Any]], filename: str, indent: int = 2) -> None:
    """
    Save data to JSON file with pretty formatting.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        List of data dictionaries to save.
    filename : str
        Output filename for the JSON file.
    indent : int, default 2
        JSON indentation level for pretty formatting.

    Raises
    ------
    Exception
        If saving fails.
    """
    if not data:
        logger.warning("No data to save")
        return

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        logger.info(f"Saved {len(data)} records to {filename}")

    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        raise
