import logging
from typing import List, Dict, Any
from PIL import Image
from ..utils.llm_interface import LLMInterface
from ..utils.prompt_loader import load_prompt
from ..models.signature_data import SignatureData

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