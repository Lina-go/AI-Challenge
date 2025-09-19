import logging
from typing import List, Dict, Any
from .pdf_processor import PDFProcessor
from .model_manager import ModelManager
from .signature_detector import SignatureDetector
from .image_processor import ImageProcessor
from .result_manager import ResultManager

logger = logging.getLogger(__name__)


class ExtractionOrchestrator:
    """
    Coordinates the complete signature extraction workflow.
    
    Single Responsibility: High-level workflow coordination.
    """
    
    def __init__(self, config: ExtractorConfig):
        self.config = config
        self.pdf_processor = PDFProcessor(config)
        self.model_manager = ModelManager(config)
        self.detector = SignatureDetector(config)
        self.image_processor = ImageProcessor(config)
        self.result_manager = ResultManager(config)
    
    def process_pdf(self, pdf_path: str, source_id: str) -> Dict[str, Any]:
        """
        Process a single PDF and extract signatures.
        
        Args:
            pdf_path: Path to PDF file
            source_id: Unique identifier for this source
            
        Returns:
            Dictionary with extraction results
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Step 1: Convert PDF to images
            page_images = self.pdf_processor.pdf_to_images(pdf_path)
            
            # Step 2: Detect signatures on each page
            detections = []
            for page_num, image_path in page_images:
                page_detections = self.detector.detect_signatures(image_path)
                for detection in page_detections:
                    detection.update({
                        'source_id': source_id,
                        'page_number': page_num,
                        'image_path': image_path
                    })
                    detections.append(detection)
            
            # Step 3: Process and crop detected signatures
            processed_results = []
            for detection in detections:
                if detection['picked']:  # Only process selected detections
                    crop_path = self.image_processor.save_signature_crop(
                        detection['image_path'],
                        detection,
                        source_id,
                        detection['page_number']
                    )
                    detection['crop_path'] = crop_path
                    processed_results.append(detection)
            
            # Step 4: Save results
            result_summary = self.result_manager.save_pdf_results(
                source_id, processed_results
            )
            
            logger.info(f"Completed {pdf_path}: {len(processed_results)} signatures")
            return result_summary
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {'source_id': source_id, 'error': str(e)}