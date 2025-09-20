# signature_extractor_v5/core/vlm_analyzer.py
"""VLM-based signature analysis using Docling."""

import logging
from typing import Dict, Any, List, Optional
from PIL import Image
import torch
from transformers import pipeline
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel import vlm_model_specs

from ..config import DoclingConfig
from ..prompts import SIGNATURE_DETECTION_PROMPT, SIGNATURE_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class VLMAnalyzer:
    """Analyzes documents using Docling VLM."""
    
    def __init__(self, config: DoclingConfig):
        self.config = config
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the Docling VLM pipeline."""
        logger.info(f"Initializing VLM pipeline with model: {self.config.model_name}")
        
        # Initialize transformers pipeline for direct VLM usage
        self.vlm_pipeline = pipeline(
            "image-text-to-text",
            model=self.config.model_name,
            device_map=self.config.device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Initialize Docling converter for document processing
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
        )
        
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options
                )
            }
        )
        
        logger.info("VLM pipeline initialized successfully")
    
    def analyze_page_for_signatures(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze a page image for signature presence.
        
        Args:
            image: PIL Image of the page
            
        Returns:
            Dictionary with signature detection results
        """
        try:
            # Prepare message for VLM
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": SIGNATURE_DETECTION_PROMPT}
                    ]
                }
            ]
            
            # Generate response
            response = self.vlm_pipeline(
                messages,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature
            )
            
            # Parse response
            return self._parse_detection_response(response[0]["generated_text"])
            
        except Exception as e:
            logger.error(f"Error in signature detection: {e}")
            return {
                "has_signatures": False,
                "error": str(e)
            }
    
    def extract_signature_data(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Extract detailed signature information from page.
        
        Args:
            image: PIL Image of the page
            
        Returns:
            List of signature data dictionaries
        """
        try:
            # Prepare message for VLM
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": SIGNATURE_EXTRACTION_PROMPT}
                    ]
                }
            ]
            
            # Generate response
            response = self.vlm_pipeline(
                messages,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature
            )
            
            # Parse response
            return self._parse_extraction_response(response[0]["generated_text"])
            
        except Exception as e:
            logger.error(f"Error in signature extraction: {e}")
            return []
    
    def _parse_detection_response(self, text: str) -> Dict[str, Any]:
        """Parse VLM detection response."""
        result = {
            "has_signatures": False,
            "signature_count": 0,
            "signature_type": "unknown",
            "is_image": False,
            "is_scanned": False,
            "confidence": "low"
        }
        
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip().lower()
                
                if 'HAS_SIGNATURES' in key:
                    result["has_signatures"] = 'yes' in value
                elif 'SIGNATURE_COUNT' in key:
                    try:
                        result["signature_count"] = int(value)
                    except:
                        pass
                elif 'SIGNATURE_TYPE' in key:
                    result["signature_type"] = value
                elif 'IS_IMAGE' in key:
                    result["is_image"] = 'yes' in value
                elif 'IS_SCANNED' in key:
                    result["is_scanned"] = 'yes' in value
                elif 'CONFIDENCE' in key:
                    result["confidence"] = value
        
        return result
    
    def _parse_extraction_response(self, text: str) -> List[Dict[str, Any]]:
        """Parse VLM extraction response."""
        signatures = []
        
        if "NO_SIGNATURES_FOUND" in text:
            return signatures
        
        current_signature = None
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('SIGNATORY_'):
                if current_signature:
                    signatures.append(current_signature)
                current_signature = {
                    "name": "",
                    "title": "",
                    "company": "",
                    "date": ""
                }
            elif current_signature and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip()
                
                if 'NAME' in key:
                    current_signature["name"] = value
                elif 'TITLE' in key:
                    current_signature["title"] = value
                elif 'COMPANY' in key:
                    current_signature["company"] = value
                elif 'DATE' in key:
                    current_signature["date"] = value
        
        if current_signature:
            signatures.append(current_signature)
        
        return signatures
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process entire PDF document using Docling.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Document analysis results
        """
        try:
            result = self.doc_converter.convert(source=pdf_path)
            doc = result.document
            
            return {
                "text": doc.export_to_markdown(),
                "page_count": len(doc.pages) if hasattr(doc, 'pages') else 1,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            }
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {"error": str(e)}