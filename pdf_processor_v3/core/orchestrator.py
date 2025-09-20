"""
Main orchestrator for unified PDF processing and signature extraction.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

from ..config import ProcessorConfig
from .document_processor import DocumentProcessor
from .signature_analyzer import SignatureAnalyzer
from .data_processor import DataProcessor
from .result_manager import ResultManager
from ..utils.progress_tracker import ProgressTracker
from ..utils.pdf_converter import convert_pdf_to_images

logger = logging.getLogger(__name__)

class MainOrchestrator:
    """Main orchestrator that handles both document processing and signature extraction"""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.document_processor = DocumentProcessor(config)
        
        # Initialize components based on task type
        if config.task.task_type == "signature_extraction" or config.task.extract_signatures:
            self.signature_analyzer = SignatureAnalyzer(config)
            self.data_processor = DataProcessor(config)
            self.result_manager = ResultManager(config)
        else:
            self.signature_analyzer = None
            self.data_processor = None
            self.result_manager = None
    
    def process_multiple_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple PDF sources with progress tracking"""
        progress = ProgressTracker(len(sources), f"Processing PDFs ({self.config.task.task_type})")
        results = []
        
        for source in sources:
            try:
                if self.config.task.task_type == "signature_extraction":
                    result = self.process_signature_extraction(source)
                else:
                    result = self.process_document_analysis(source)
                    
                results.append(result)
                progress.update()
            except Exception as e:
                logger.error(f"Failed to process {source.get('source_id', 'unknown')}: {e}")
                progress.update(error=True)
        
        # Save consolidated results if doing signature extraction
        if self.config.task.task_type == "signature_extraction" and self.result_manager:
            self.result_manager.save_consolidated_results(results)
        
        return results
    
    def process_signature_extraction(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single PDF source for signature extraction"""
        pdf_path = source['pdf_path']
        source_id = source['source_id']
        
        logger.info(f"Extracting signatures from {source_id}: {pdf_path}")
        
        # Step 1: Convert PDF to images
        page_images = convert_pdf_to_images(pdf_path)
        
        # Step 2: Analyze each page for signatures
        signature_results = []
        for page_num, image in page_images:
            # Save image temporarily for analysis
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image.save(tmp_file.name, 'PNG')
                temp_image_path = tmp_file.name
            
            try:
                page_signatures = self.signature_analyzer.analyze_page(temp_image_path, page_num)
                for signature in page_signatures:
                    signature.update({
                        'source_id': source_id,
                        'pdf_path': pdf_path,
                        'page_number': page_num
                    })
                    signature_results.append(signature)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_image_path)
                except OSError:
                    pass
        
        # Step 3: Process and clean extracted data
        processed_results = self.data_processor.process_signatures(signature_results)
        
        # Step 4: Save results for this source
        return self.result_manager.save_source_results(source_id, processed_results)
    
    def process_document_analysis(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single PDF source for full document analysis"""
        pdf_path = source['pdf_path']
        source_id = source['source_id']
        
        logger.info(f"Processing document {source_id}: {pdf_path}")
        
        # Use the existing DocumentProcessor
        result = self.document_processor.process_document(pdf_path)
        
        # Save results to output directory
        output_file = Path(self.config.processing.output_dir) / f"{source_id}_content.md"
        self.document_processor.save_results(result, str(output_file))
        
        # Return summary
        return {
            'source_id': source_id,
            'pdf_path': pdf_path,
            'total_pages': result.total_pages,
            'processed_pages': result.processed_pages,
            'skipped_pages': result.skipped_pages,
            'success_rate': result.get_success_rate(),
            'output_file': str(output_file),
            'final_markdown_length': len(result.final_markdown or '')
        }


# Convenience classes for backward compatibility
class SignatureOrchestrator(MainOrchestrator):
    """Specialized orchestrator for signature extraction (backward compatibility)"""
    
    def __init__(self, config: ProcessorConfig):
        # Ensure signature extraction is enabled
        config.task.task_type = "signature_extraction"
        config.task.extract_signatures = True
        super().__init__(config)
    
    def process_multiple_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple sources for signature extraction"""
        return super().process_multiple_sources(sources)


class DocumentOrchestrator(MainOrchestrator):
    """Specialized orchestrator for document processing (backward compatibility)"""
    
    def __init__(self, config: ProcessorConfig):
        # Ensure document processing is enabled
        config.task.task_type = "document_processing"
        config.task.extract_signatures = False
        super().__init__(config)
    
    def process_multiple_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple sources for document analysis"""
        return super().process_multiple_sources(sources)