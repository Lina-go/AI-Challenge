# signature_extractor_v5/core/result_formatter.py
"""Format results to match v3 output structure."""

import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultFormatter:
    """Formats extraction results to match required output."""
    
    def format_signature_results(
        self,
        detection_results: List[Dict[str, Any]],
        extraction_results: List[List[Dict[str, Any]]],
        source_id: str
    ) -> List[Dict[str, Any]]:
        """
        Format results to match v3 output columns.
        
        Args:
            detection_results: List of detection results per page
            extraction_results: List of extraction results per page
            source_id: Source document identifier
            
        Returns:
            Formatted results matching required output structure
        """
        formatted_results = []
        
        for page_num, (detection, extraction) in enumerate(
            zip(detection_results, extraction_results), 1
        ):
            if not detection.get("has_signatures", False):
                # No signature on this page
                if page_num == 1:  # Add one "No" record per document
                    formatted_results.append({
                        "Signature": "No",
                        "Signature_Yes_text": "",
                        "Signature_Image": "No",
                        "Signature_scanned": "No",
                        "Presence_Signature_date": "No",
                        "Signature_Date": "",
                        "source_id": source_id,
                        "page_number": page_num,
                        "confidence": "N/A",
                        "processing_timestamp": datetime.now().isoformat()
                    })
            else:
                # Process each signature found
                for sig in extraction:
                    formatted_results.append({
                        "Signature": "Yes",
                        "Signature_Yes_text": self._format_signatory_text(sig),
                        "Signature_Image": "Yes" if detection.get("is_image") else "No",
                        "Signature_scanned": "Yes" if detection.get("is_scanned") else "No",
                        "Presence_Signature_date": "Yes" if sig.get("date") else "No",
                        "Signature_Date": self._format_date(sig.get("date", "")),
                        "source_id": source_id,
                        "page_number": page_num,
                        "confidence": detection.get("confidence", "medium"),
                        "processing_timestamp": datetime.now().isoformat()
                    })
        
        return formatted_results
    
    def _format_signatory_text(self, sig: Dict[str, Any]) -> str:
        """Format signatory information."""
        parts = []
        if sig.get("name"):
            parts.append(sig["name"])
        if sig.get("title"):
            parts.append(sig["title"])
        if sig.get("company"):
            parts.append(sig["company"])
        return " | ".join(parts)
    
    def _format_date(self, date_str: str) -> str:
        """Format date string."""
        if not date_str:
            return ""
        
        # Try to standardize to YYYY-MM-DD format
        try:
            from dateutil import parser
            parsed = parser.parse(date_str)
            return parsed.strftime("%Y-%m-%d")
        except:
            return date_str