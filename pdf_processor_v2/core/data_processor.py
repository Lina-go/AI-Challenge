import logging
import re
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class DataProcessor:
    """Processes and cleans extracted signature data"""
    
    def __init__(self, config):
        self.config = config
    
    def process_signatures(self, raw_signatures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and clean raw signature extraction results
        
        Args:
            raw_signatures: List of raw signature data from analyzer
            
        Returns:
            List of processed signature data matching output format
        """
        processed_results = []
        
        for signature in raw_signatures:
            try:
                processed = self._process_single_signature(signature)
                processed_results.append(processed)
            except Exception as e:
                logger.error(f"Error processing signature: {e}")
                # Add error record
                processed_results.append(self._create_error_record(signature, str(e)))
        
        return processed_results
    
    def _process_single_signature(self, signature: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single signature record"""
        
        # Extract and clean signatory text
        signatory_text = self._clean_signatory_text(signature.get('signatory_text', ''))
        
        # Process date information
        date_info = self._process_date_info(signature)
        
        # Create final processed record
        processed = {
            # Basic signature presence
            'Signature': 'Yes' if signature.get('has_signature', False) else 'No',
            
            # Signatory information
            'Signature_Yes_text': signatory_text,
            
            # Signature type information
            'Signature_Image': 'Yes' if signature.get('is_image_signature', False) else 'No',
            'Signature_scanned': 'Yes' if signature.get('is_scanned', False) else 'No',
            
            # Date information
            'Presence_Signature_date': 'Yes' if signature.get('has_date', False) else 'No',
            'Signature_Date': date_info.get('formatted_date', ''),
            
            # Metadata
            'source_id': signature.get('source_id', ''),
            'page_number': signature.get('page_number', 0),
            'confidence': signature.get('confidence', 'low'),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return processed
    
    def _clean_signatory_text(self, text: str) -> str:
        """Clean and normalize signatory text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.,\-\(\)\/]', '', text)
        
        return text
    
    def _process_date_info(self, signature: Dict[str, Any]) -> Dict[str, str]:
        """Process and format date information"""
        if not signature.get('has_date', False):
            return {'formatted_date': ''}
        
        raw_date = signature.get('signature_date', '')
        if not raw_date:
            return {'formatted_date': ''}
        
        # Try to standardize date format
        try:
            # If already in YYYY-MM-DD format
            if re.match(r'\d{4}-\d{2}-\d{2}', raw_date):
                return {'formatted_date': raw_date}
            
            # Try common date formats
            date_formats = [
                '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
                '%m-%d-%Y', '%d-%m-%Y', '%Y-%m-%d',
                '%B %d, %Y', '%d %B %Y'
            ]
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(raw_date, fmt)
                    return {'formatted_date': parsed_date.strftime('%Y-%m-%d')}
                except ValueError:
                    continue
            
            # If no format matches, return as-is
            return {'formatted_date': raw_date}
            
        except Exception as e:
            logger.warning(f"Error processing date '{raw_date}': {e}")
            return {'formatted_date': raw_date}
    
    def _create_error_record(self, signature: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Create an error record for failed processing"""
        return {
            'Signature': 'Error',
            'Signature_Yes_text': f'Processing Error: {error_msg}',
            'Signature_Image': 'No',
            'Signature_scanned': 'No',
            'Presence_Signature_date': 'No',
            'Signature_Date': '',
            'source_id': signature.get('source_id', ''),
            'page_number': signature.get('page_number', 0),
            'confidence': 'error',
            'processing_timestamp': datetime.now().isoformat()
        }