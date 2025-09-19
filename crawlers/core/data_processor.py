"""
Data processing utilities for web crawling.
"""

import logging
from typing import List, Dict, Any, Optional

from ..utils import clean_extracted_data

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Responsible for processing, cleaning, and enriching extracted data.

    Single Responsibility: Data transformation and validation.
    """

    def __init__(self, jurisdiction_code: str):
        """
        Initialize data processor.

        Parameters
        ----------
        jurisdiction_code : str
            Code for the jurisdiction (e.g., "CA", "GB").
        """
        self.jurisdiction_code = jurisdiction_code

    def process_statement_data(
        self, raw_data: Dict[str, Any], source_url: str
    ) -> Optional[Dict[str, Any]]:
        """
        Process and enrich a single statement's data.

        Parameters
        ----------
        raw_data : Dict[str, Any]
            Raw extracted data from the statement page.
        source_url : str
            URL where the data was extracted from.

        Returns
        -------
        Dict[str, Any] or None
            Processed and enriched data dictionary if valid, None if invalid.
        """
        # Clean the raw data
        cleaned_data = clean_extracted_data(raw_data)

        if not cleaned_data:
            return None

        # Enrich with metadata
        cleaned_data["source_url"] = source_url
        cleaned_data["jurisdiction"] = self.jurisdiction_code

        return cleaned_data

    def process_batch(self, raw_data_list: List[tuple]) -> List[Dict[str, Any]]:
        """
        Process a batch of raw data entries.

        Parameters
        ----------
        raw_data_list : List[tuple]
            List of (raw_data, source_url) tuples.

        Returns
        -------
        List[Dict[str, Any]]
            List of processed data dictionaries.
        """
        processed_data = []

        for raw_data, source_url in raw_data_list:
            processed = self.process_statement_data(raw_data, source_url)
            if processed:
                processed_data.append(processed)

        return processed_data

    def validate_required_fields(
        self, data: Dict[str, Any], required_fields: List[str]
    ) -> bool:
        """
        Validate that required fields are present and non-empty.

        Parameters
        ----------
        data : Dict[str, Any]
            Data dictionary to validate.
        required_fields : List[str]
            List of required field names.

        Returns
        -------
        bool
            True if all required fields are present and valid.
        """
        for field in required_fields:
            if field not in data or not data[field]:
                logger.debug(f"Missing or empty required field: {field}")
                return False
        return True

    def calculate_statistics(
        self, successful_count: int, failed_count: int
    ) -> Dict[str, Any]:
        """
        Calculate crawling statistics.

        Parameters
        ----------
        successful_count : int
            Number of successful extractions.
        failed_count : int
            Number of failed extractions.

        Returns
        -------
        Dict[str, Any]
            Dictionary with statistics.
        """
        total = successful_count + failed_count
        success_rate = (successful_count / total * 100) if total > 0 else 0

        return {
            "successful_extractions": successful_count,
            "failed_extractions": failed_count,
            "total_processed": total,
            "success_rate": success_rate,
        }
