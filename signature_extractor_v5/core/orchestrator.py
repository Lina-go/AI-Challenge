# signature_extractor_v5/core/orchestrator.py
"""Main orchestration of extraction workflow."""

import logging
from typing import List, Dict, Any
from pathlib import Path

from ..config import ExtractionConfig
from .document_processor import DocumentProcessor
from .vlm_analyzer import VLMAnalyzer
from .result_formatter import ResultFormatter

logger = logging.getLogger(__name__)


class ExtractionOrchestrator:
    """Coordinates the extraction workflow."""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.doc_processor = DocumentProcessor(config.processing.output_dir)
        self.analyzer = VLMAnalyzer(config.docling)
        self.formatter = ResultFormatter()
    
    def process_document(self, pdf_path: str, source_id: str) -> List[Dict[str, Any]]:
        """
        Process a single PDF document.
        
        Args:
            pdf_path: Path to PDF file
            source_id: Source identifier
            
        Returns:
            List of formatted signature results
        """
        logger.info(f"Processing document: {source_id}")
        
        try:
            # Extract pages as images
            pages = self.doc_processor.extract_pages_as_images(
                pdf_path,
                max_pages=self.config.processing.max_pages
            )
            
            detection_results = []
            extraction_results = []
            
            # Process each page
            for page_num, image in pages:
                logger.debug(f"Analyzing page {page_num}")
                
                # Detect signatures
                detection = self.analyzer.analyze_page_for_signatures(image)
                detection_results.append(detection)
                
                # Extract signature data if signatures found
                if detection.get("has_signatures"):
                    extraction = self.analyzer.extract_signature_data(image)
                    extraction_results.append(extraction)
                else:
                    extraction_results.append([])
                
                # Optionally save intermediate images
                if self.config.processing.save_intermediate:
                    self.doc_processor.save_page_image(
                        image, 
                        Path(pdf_path).stem,
                        page_num
                    )
            
            # Format results
            results = self.formatter.format_signature_results(
                detection_results,
                extraction_results,
                source_id
            )
            
            logger.info(f"Completed processing {source_id}: {len(results)} signatures")
            return results
            
        except Exception as e:
            logger.error(f"Error processing {source_id}: {e}")
            return [{
                "Signature": "Error",
                "Signature_Yes_text": str(e),
                "Signature_Image": "No",
                "Signature_scanned": "No",
                "Presence_Signature_date": "No",
                "Signature_Date": "",
                "source_id": source_id,
                "page_number": 0,
                "confidence": "error",
                "processing_timestamp": datetime.now().isoformat()
            }]
    
    def process_batch(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents.
        
        Args:
            sources: List of source dictionaries with pdf_path and source_id
            
        Returns:
            Combined results from all documents
        """
        all_results = []
        
        for source in sources:
            results = self.process_document(
                source["pdf_path"],
                source["source_id"]
            )
            all_results.extend(results)
        
        return all_results