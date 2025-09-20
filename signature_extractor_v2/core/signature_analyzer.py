import logging
import re
from typing import List, Dict, Any
from PIL import Image
from ..utils.llm_interface import LLMInterface
from ..utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

class SignatureAnalyzer:
    """LLM-based signature detection and analysis"""
    
    def __init__(self, config):
        self.config = config
        self.llm = LLMInterface(config.llm)
    
    def analyze_page(self, image_path: str, page_number: int) -> List[Dict[str, Any]]:
        """Analyze a page for signatures and extract metadata"""
        try:
            # Step 1: Detect if signatures are present
            detection_result = self._detect_signatures(image_path)
            
            if not detection_result.get('has_signatures', False):
                return []
            
            # Step 2: Analyze signature details
            signature_analysis = self._analyze_signature_details(image_path)
            
            # Step 3: Extract signatory text and dates
            text_data = self._extract_signatory_text(image_path)
            date_data = self._extract_signature_dates(image_path)
            
            # Step 4: Combine results into structured format
            return self._combine_analysis_results(
                detection_result, signature_analysis, text_data, date_data, page_number
            )
            
        except Exception as e:
            logger.error(f"Error analyzing page {page_number}: {e}")
            return []
    
    def _detect_signatures(self, image_path: str) -> Dict[str, Any]:
        """Detect presence of signatures on page"""
        prompt = load_prompt("signature_detector")
        image = Image.open(image_path)
        response = self.llm.process_image_with_prompt(image, prompt)
        return self._parse_detection_response(response)
    
    def _analyze_signature_details(self, image_path: str) -> Dict[str, Any]:
        """Analyze signature characteristics (image, scanned, etc.)"""
        prompt = load_prompt("signature_analyzer")
        image = Image.open(image_path)
        response = self.llm.process_image_with_prompt(image, prompt)
        return self._parse_analysis_response(response)
    
    def _extract_signatory_text(self, image_path: str) -> Dict[str, Any]:
        """Extract signatory names, titles, and related text"""
        prompt = load_prompt("text_extractor")
        image = Image.open(image_path)
        response = self.llm.process_image_with_prompt(image, prompt)
        return self._parse_text_response(response)
    
    def _extract_signature_dates(self, image_path: str) -> Dict[str, Any]:
        """Extract dates associated with signatures"""
        prompt = load_prompt("date_extractor")
        image = Image.open(image_path)
        response = self.llm.process_image_with_prompt(image, prompt)
        return self._parse_date_response(response)
    
    def _parse_detection_response(self, response: str) -> Dict[str, Any]:
        """Parse signature detection response"""
        result = {
            'has_signatures': False,
            'signature_count': 0,
            'confidence': 'low'
        }
        
        try:
            lines = response.lower().split('\n')
            for line in lines:
                if 'has_signatures:' in line:
                    result['has_signatures'] = 'true' in line
                elif 'signature_count:' in line:
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        result['signature_count'] = int(numbers[0])
                elif 'confidence:' in line:
                    if 'high' in line:
                        result['confidence'] = 'high'
                    elif 'medium' in line:
                        result['confidence'] = 'medium'
        except Exception as e:
            logger.error(f"Error parsing detection response: {e}")
        
        return result
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse signature analysis response"""
        result = {
            'signature_type': 'unknown',
            'is_image': False,
            'is_scanned': False
        }
        
        try:
            lines = response.lower().split('\n')
            for line in lines:
                if 'signature_type:' in line:
                    if 'handwritten' in line:
                        result['signature_type'] = 'handwritten'
                    elif 'typed' in line:
                        result['signature_type'] = 'typed'
                elif 'is_image:' in line:
                    result['is_image'] = 'true' in line
                elif 'is_scanned:' in line:
                    result['is_scanned'] = 'true' in line
        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}")
        
        return result
    
    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """Parse signatory text response"""
        result = {
            'signatory_text': '',
            'extracted_names': [],
            'extracted_titles': [],
            'extracted_companies': []
        }
        
        try:
            # Look for SIGNATORY_TEXT_1: pattern
            if 'SIGNATORY_TEXT_1:' in response:
                text_start = response.find('SIGNATORY_TEXT_1:') + len('SIGNATORY_TEXT_1:')
                text_end = response.find('SIGNATORY_TEXT_2:', text_start)
                if text_end == -1:
                    text_end = len(response)
                
                signatory_text = response[text_start:text_end].strip()
                result['signatory_text'] = signatory_text
            else:
                # Fallback: use the whole response
                result['signatory_text'] = response.strip()
        
        except Exception as e:
            logger.error(f"Error parsing text response: {e}")
        
        return result
    
    def _parse_date_response(self, response: str) -> Dict[str, Any]:
        """Parse date extraction response"""
        result = {
            'has_date': False,
            'signature_date': '',
            'raw_date_text': ''
        }
        
        try:
            lines = response.split('\n')
            for line in lines:
                if 'raw_date_text:' in line:
                    date_text = line.split('raw_date_text:')[1].strip()
                    result['raw_date_text'] = date_text
                    if date_text:
                        result['has_date'] = True
                elif 'standardized_date:' in line:
                    std_date = line.split('standardized_date:')[1].strip()
                    if std_date and std_date != '[YYYY-MM-DD format if determinable]':
                        result['signature_date'] = std_date
        
        except Exception as e:
            logger.error(f"Error parsing date response: {e}")
        
        return result
    
    def _combine_analysis_results(self, detection, analysis, text, date, page_number) -> List[Dict[str, Any]]:
        """Combine all analysis results into final format"""
        if not detection.get('has_signatures', False):
            return []
        
        # Create a single signature result (can be expanded for multiple signatures)
        signature_result = {
            'has_signature': True,
            'signatory_text': text.get('signatory_text', ''),
            'is_image_signature': analysis.get('is_image', False),
            'is_scanned': analysis.get('is_scanned', False),
            'has_date': date.get('has_date', False),
            'signature_date': date.get('signature_date', ''),
            'page_number': page_number,
            'confidence': detection.get('confidence', 'low')
        }
        
        return [signature_result]