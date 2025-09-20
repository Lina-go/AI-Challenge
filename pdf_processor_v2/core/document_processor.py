"""
Document-level processing logic for PDF pipeline with integrated signature extraction.
"""

import logging
from pathlib import Path
from typing import Optional

from .llm_interface import LLMInterface, create_llm_interface
from .page_processor import PageProcessor
from ..models.document_result import DocumentResult
from ..utils.pdf_converter import convert_pdf_to_images, get_pdf_info
from ..utils.content_integrator import combine_page_contents

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles processing of complete PDF documents with integrated signature extraction."""
    
    def __init__(self, config, debug_dir: Optional[str] = None):
        """
        Initialize document processor.
        
        Args:
            config: ProcessorConfig object with LLM and processing configuration
            debug_dir: Optional directory for debug outputs
        """
        self.config = config
        self.llm = create_llm_interface(config.llm)
        
        # Pass config to PageProcessor for signature extraction capability
        self.page_processor = PageProcessor(self.llm, config, debug_dir)
    
    def process_document(self, pdf_path: str, output_dir: Optional[str] = None) -> DocumentResult:
        """
        Process a complete PDF document through the full pipeline with integrated signature extraction.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Optional directory for saving intermediate results
            
        Returns:
            DocumentResult with all processing results
        """
        pdf_path = str(Path(pdf_path).resolve())
        logger.info(f"Starting processing of PDF: {pdf_path}")
        
        try:
            # Get PDF information
            pdf_info = get_pdf_info(pdf_path)
            total_pages = pdf_info["page_count"]
            
            logger.info(f"PDF has {total_pages} pages")
            
            # Initialize result
            result = DocumentResult(
                pdf_path=pdf_path,
                total_pages=total_pages
            )
            
            logger.info("Converting PDF to images...")
            page_images = convert_pdf_to_images(pdf_path)
            
            processed_contents = []
            signature_count = 0
            
            for page_number, image in page_images:
                logger.info(f"Processing page {page_number}/{total_pages}")
                
                try:
                    page_content = self.page_processor.process_page(image, page_number)
                    
                    if page_content is not None:
                        result.add_page_content(page_content)
                        processed_contents.append(page_content)
                        
                        # Count signatures found
                        if page_content.signature_mapping:
                            signature_count += len(page_content.signature_mapping)
                            
                    else:
                        result.skipped_pages += 1
                        logger.info(f"Skipped page {page_number} (no meaningful content)")
                
                except Exception as e:
                    logger.error(f"Failed to process page {page_number}: {e}")
                    result.skipped_pages += 1
            
            if processed_contents:
                result.final_markdown = combine_page_contents(processed_contents)
                logger.info(f"Successfully processed {result.processed_pages} pages, skipped {result.skipped_pages}")
                if signature_count > 0:
                    logger.info(f"Extracted signature information from {signature_count} signature blocks")
            else:
                logger.warning("No content was successfully extracted from any page")
                result.final_markdown = ""
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return DocumentResult(
                pdf_path=pdf_path,
                total_pages=0,
                final_markdown=f"Processing failed: {e}"
            )
    
    def save_results(self, result: DocumentResult, output_path: str) -> None:
        """
        Save processing results to file.
        
        Args:
            result: DocumentResult to save
            output_path: Path where to save the markdown file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if result.final_markdown:
                    f.write(result.final_markdown)
                else:
                    f.write("No content was extracted from the document.")
            
            logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def get_processing_summary(self, result: DocumentResult) -> str:
        """
        Generate a human-readable processing summary with signature information.
        
        Args:
            result: DocumentResult to summarize
            
        Returns:
            Summary string
        """
        stats = result.processing_stats
        success_rate = result.get_success_rate()
        
        # Count signatures in the final markdown
        signature_sections = 0
        if result.final_markdown:
            signature_sections = len([line for line in result.final_markdown.split('\n') if 'Signature Information' in line])
        
        summary = f"""
PDF Processing Summary (with Signature Extraction)
=================================================
File: {result.pdf_path}
Total Pages: {result.total_pages}
Processed Pages: {result.processed_pages}
Skipped Pages: {result.skipped_pages}
Success Rate: {success_rate:.1f}%

Content Statistics:
- Total Tables: {stats['total_tables']}
- Total Figures: {stats['total_figures']}
- Total Signatures Extracted: {signature_sections}
- Total Placeholders: {stats['total_placeholders']}
- Successful Integrations: {stats['successful_integrations']}

Final Document Length: {len(result.final_markdown or '') // 1000}K characters

Signature Extraction: {'Enabled' if self.config else 'Disabled'}
        """.strip()
        
        return summary