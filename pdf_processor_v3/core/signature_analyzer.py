import logging
import re
from typing import List, Dict, Any
from PIL import Image
from .llm_interface import create_llm_interface
from ..utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

class SignatureAnalyzer:
    """LLM-based signature detection and analysis"""
    
    def __init__(self, config):
        self.config = config
        self.llm = create_llm_interface(config.llm)
    
    def analyze_page(self, image_path: str, page_number: int) -> List[Dict[str, Any]]:
        """Analyze a page for signatures using optimized 2-prompt approach"""
        try:
            # STEP 1: Signature detection and characteristic analysis
            detection_result = self._signature_detection_and_analysis(image_path)
            
            if not detection_result.get('has_signatures', False):
                return []
            
            # STEP 2: Complete data extraction
            extraction_result = self._signature_data_extraction(image_path)
            
            # STEP 3: Combine and format results
            return self._format_final_results(detection_result, extraction_result, page_number)
            
        except Exception as e:
            logger.error(f"Error analyzing page {page_number}: {e}")
            return []
    
    def _signature_detection_and_analysis(self, image_path: str) -> Dict[str, Any]:
        """PROMPT 1: Signature detection and characteristic analysis"""
        prompt = load_prompt("signature_detector_analyzer")
        image = Image.open(image_path)
        response = self.llm.process_image_with_prompt(image, prompt)
        return self._parse_detection_and_analysis_response(response)
    
    def _signature_data_extraction(self, image_path: str) -> Dict[str, Any]:
        """PROMPT 2: Complete signature data extraction"""
        prompt = load_prompt("signature_data_extractor")
        image = Image.open(image_path)
        response = self.llm.process_image_with_prompt(image, prompt)
        return self._parse_data_extraction_response(response)
    
    def _parse_detection_and_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse signature detection and analysis response"""
        result = {
            'has_signatures': False, 
            'signature_count': 0, 
            'confidence': 'low',
            'signatures_analysis': []
        }
        
        lines = response.split('\n')
        current_signature = None
        
        for line in lines:
            line = line.strip()
            line_lower = line.lower()
            
            # Parse detection section
            if 'has_signatures:' in line_lower:
                result['has_signatures'] = 'true' in line_lower
            elif 'signature_count:' in line_lower:
                numbers = re.findall(r'\d+', line)
                if numbers:
                    result['signature_count'] = int(numbers[0])
            elif 'confidence:' in line_lower:
                if 'high' in line_lower:
                    result['confidence'] = 'high'
                elif 'medium' in line_lower:
                    result['confidence'] = 'medium'
            
            # Parse signature analysis section
            elif line.startswith('SIGNATURE_') and ':' in line:
                if current_signature:
                    result['signatures_analysis'].append(current_signature)
                current_signature = {
                    'signature_type': 'unknown',
                    'is_image': False,
                    'is_scanned': False,
                    'signature_block_format': 'none'
                }
            elif current_signature and ':' in line:
                if 'signature_type:' in line_lower:
                    sig_type = line.split(':')[1].strip().lower()
                    current_signature['signature_type'] = sig_type
                elif 'is_image:' in line_lower:
                    current_signature['is_image'] = 'true' in line_lower
                elif 'is_scanned:' in line_lower:
                    current_signature['is_scanned'] = 'true' in line_lower
                elif 'signature_block_format:' in line_lower:
                    format_type = line.split(':')[1].strip().lower()
                    current_signature['signature_block_format'] = format_type
        
        # Add the last signature if exists
        if current_signature:
            result['signatures_analysis'].append(current_signature)
        
        return result
    
    def _parse_data_extraction_response(self, response: str) -> Dict[str, Any]:
        """Parse signature data extraction response"""
        result = {
            'signatory_texts': [],
            'dates': [],
            'has_date': False
        }
        
        lines = response.split('\n')
        current_section = None
        current_text = []
        current_date = None
        
        for line in lines:
            line_original = line.rstrip()
            line_lower = line.lower().strip()
            
            # Section headers
            if line_lower.startswith('signatory_text_'):
                if current_text:
                    result['signatory_texts'].append('\n'.join(current_text))
                current_text = []
                current_section = 'signatory_text'
                continue
            elif line_lower.startswith('date_'):
                if current_text and current_section == 'signatory_text':
                    result['signatory_texts'].append('\n'.join(current_text))
                    current_text = []
                if current_date:
                    result['dates'].append(current_date)
                current_date = {
                    'raw_date_text': '',
                    'associated_signature': '',
                    'standardized_date': '',
                    'date_confidence': 'low',
                    'position_context': ''
                }
                current_section = 'date'
                continue
            
            # Content parsing
            if current_section == 'signatory_text' and line_original and not line_lower.startswith('-'):
                current_text.append(line_original)
            elif current_section == 'date' and ':' in line:
                if 'raw_date_text:' in line_lower:
                    current_date['raw_date_text'] = line.split(':', 1)[1].strip()
                    if current_date['raw_date_text']:
                        result['has_date'] = True
                elif 'associated_signature:' in line_lower:
                    current_date['associated_signature'] = line.split(':', 1)[1].strip()
                elif 'standardized_date:' in line_lower:
                    current_date['standardized_date'] = line.split(':', 1)[1].strip()
                elif 'date_confidence:' in line_lower:
                    current_date['date_confidence'] = line.split(':', 1)[1].strip()
                elif 'position_context:' in line_lower:
                    current_date['position_context'] = line.split(':', 1)[1].strip()
        
        # Add remaining content
        if current_text and current_section == 'signatory_text':
            result['signatory_texts'].append('\n'.join(current_text))
        if current_date:
            result['dates'].append(current_date)
        
        return result
    
    def _format_final_results(self, detection: Dict, extraction: Dict, page_number: int) -> List[Dict[str, Any]]:
        """Format final signature results"""
        if not detection.get('has_signatures', False):
            return []
        
        # Combine signatory texts into a single string
        signatory_text = ''
        if extraction.get('signatory_texts'):
            signatory_text = '\n\n'.join(extraction['signatory_texts'])
        
        # Get signature characteristics from first signature analysis (or defaults)
        signature_analysis = detection.get('signatures_analysis', [{}])
        first_sig = signature_analysis[0] if signature_analysis else {}
        
        # Get date information
        dates = extraction.get('dates', [])
        signature_date = ''
        has_date = extraction.get('has_date', False)
        
        if dates:
            # Use the first date found
            first_date = dates[0]
            signature_date = first_date.get('standardized_date', '') or first_date.get('raw_date_text', '')
            has_date = bool(signature_date)
        
        signature_result = {
            'has_signature': True,
            'signatory_text': signatory_text,
            'is_image_signature': first_sig.get('is_image', False),
            'is_scanned': first_sig.get('is_scanned', False),
            'has_date': has_date,
            'signature_date': signature_date,
            'page_number': page_number,
            'confidence': detection.get('confidence', 'low')
        }
        
        return [signature_result]