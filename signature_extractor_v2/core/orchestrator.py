import logging
from typing import List, Dict, Any
from ..config import ExtractionConfig
from .document_processor import DocumentProcessor
from .signature_analyzer import SignatureAnalyzer
from .data_processor import DataProcessor
from .result_manager import ResultManager
from ..utils.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

class ExtractionOrchestrator:
    """Coordinates the complete signature extraction workflow"""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.document_processor = DocumentProcessor(config)
        self.signature_analyzer = SignatureAnalyzer(config)
        self.data_processor = DataProcessor(config)
        self.result_manager = ResultManager(config)
    
    def process_multiple_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple PDF sources with progress tracking"""
        progress = ProgressTracker(len(sources), "Extracting signatures")
        results = []
        
        for source in sources:
            try:
                result = self.process_single_source(source)
                results.append(result)
                progress.update()
            except Exception as e:
                logger.error(f"Failed to process {source.get('source_id', 'unknown')}: {e}")
                progress.update(error=True)
        
        # Save final consolidated results
        self.result_manager.save_consolidated_results(results)
        return results
    
    def process_single_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single PDF source"""
        pdf_path = source['pdf_path']
        source_id = source['source_id']
        
        logger.info(f"Processing {source_id}: {pdf_path}")
        
        # Step 1: Convert PDF to images
        page_images = self.document_processor.process_document(pdf_path)
        
        # Step 2: Analyze each page for signatures
        signature_results = []
        for page_num, image_path in page_images:
            page_signatures = self.signature_analyzer.analyze_page(image_path, page_num)
            for signature in page_signatures:
                signature.update({
                    'source_id': source_id,
                    'pdf_path': pdf_path,
                    'page_number': page_num
                })
                signature_results.append(signature)
        
        # Step 3: Process and clean extracted data
        processed_results = self.data_processor.process_signatures(signature_results)
        
        # Step 4: Save results for this source
        return self.result_manager.save_source_results(source_id, processed_results)