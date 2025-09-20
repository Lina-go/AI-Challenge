"""
Page-level processing logic for PDF pipeline.
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import Optional
from PIL import Image

from .llm_interface import LLMInterface
from .signature_analyzer import SignatureAnalyzer
from ..models.page_analysis import PageAnalysis
from ..models.page_content import PageContent
from ..utils.prompt_loader import load_prompt
from ..utils.content_parser import (
    parse_page_analysis, 
    parse_table_content, 
    parse_figure_content, 
    parse_signature_content,
    parse_signature_extraction_results
)
from ..utils.content_integrator import integrate_page_content

logger = logging.getLogger(__name__)


class PageProcessor:
    """Handles processing of individual PDF pages."""
    
    def __init__(self, llm_interface: LLMInterface, config=None, debug_dir: Optional[str] = None):
        """
        Initialize page processor.
        
        Args:
            llm_interface: LLM interface for processing images
            config: ProcessorConfig object (needed for signature analysis)
            debug_dir: Optional directory to save debug outputs
        """
        self.llm = llm_interface
        self.config = config
        self.debug_dir = debug_dir or "pdf_processor_v3/debug_logs"
        
        # Initialize signature analyzer if config provided
        self.signature_analyzer = None
        if config:
            self.signature_analyzer = SignatureAnalyzer(config)
        
        Path(self.debug_dir).mkdir(parents=True, exist_ok=True)
    
    def _save_debug_output(self, page_number: int, prompt_type: str, response: str) -> None:
        """Save prompt response to debug file."""
        try:
            debug_file = Path(self.debug_dir) / f"page_{page_number:02d}_{prompt_type}.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"=== PAGE {page_number} - {prompt_type.upper()} PROMPT OUTPUT ===\n\n")
                f.write(response)
            logger.debug(f"Saved debug output: {debug_file}")
        except Exception as e:
            logger.warning(f"Failed to save debug output: {e}")
    
    def analyze_page(self, image: Image.Image, page_number: int) -> PageAnalysis:
        """
        Analyze a page to determine content types and processing needs.
        
        Args:
            image: PIL Image of the page
            page_number: Page number (1-based)
            
        Returns:
            PageAnalysis object with analysis results
        """
        try:
            prompt = load_prompt("page_content_analyzer")
            response = self.llm.process_image_with_prompt(image, prompt)
            
            self._save_debug_output(page_number, "analysis", response)
            
            return parse_page_analysis(response, page_number)
        except Exception as e:
            return PageAnalysis(
                page_number=page_number,
                should_parse=True,
                has_text=True,
                has_tables=False,
                has_figures=False,
                has_signatures=False,
                raw_analysis=f"Analysis failed: {e}"
            )
    
    def extract_text_content(self, image: Image.Image, page_number: int) -> str:
        """
        Extract text content with placeholders from a page.
        
        Args:
            image: PIL Image of the page
            page_number: Page number for debug logging
            
        Returns:
            Text content with placeholders for tables/figures
        """
        try:
            prompt = load_prompt("text_extractor")
            response = self.llm.process_image_with_prompt(image, prompt)
            
            self._save_debug_output(page_number, "text", response)
            
            return response
        except Exception as e:
            return f"Text extraction failed: {e}"
    
    def extract_table_content(self, image: Image.Image, page_number: int) -> str:
        """
        Extract table content from a page.
        
        Args:
            image: PIL Image of the page
            page_number: Page number for debug logging
            
        Returns:
            Raw table extraction response
        """
        try:
            prompt = load_prompt("table_extractor")
            response = self.llm.process_image_with_prompt(image, prompt)
            
            self._save_debug_output(page_number, "tables", response)
            
            return response
        except Exception as e:
            return f"Table extraction failed: {e}"
    
    def extract_figure_content(self, image: Image.Image, page_number: int) -> str:
        """
        Extract figure content from a page.
        
        Args:
            image: PIL Image of the page
            page_number: Page number for debug logging
            
        Returns:
            Raw figure extraction response
        """
        try:
            prompt = load_prompt("figure_extractor")
            response = self.llm.process_image_with_prompt(image, prompt)
            
            self._save_debug_output(page_number, "figures", response)
            
            return response
        except Exception as e:
            return f"Figure extraction failed: {e}"
    
    def extract_signature_content(self, image: Image.Image, page_number: int) -> list:
        """
        Extract signature content from a page using SignatureAnalyzer.
        
        Args:
            image: PIL Image of the page
            page_number: Page number for debug logging
            
        Returns:
            List of signature extraction results
        """
        if not self.signature_analyzer:
            logger.warning(f"Signature analyzer not available for page {page_number}")
            return []
        
        try:
            # Save image temporarily for signature analysis
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image.save(tmp_file.name, 'PNG')
                temp_image_path = tmp_file.name
            
            try:
                # Analyze signatures on this page
                signature_results = self.signature_analyzer.analyze_page(temp_image_path, page_number)
                
                self._save_debug_output(page_number, "signatures", str(signature_results))
                
                logger.debug(f"Extracted {len(signature_results)} signatures from page {page_number}")
                return signature_results
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_image_path)
                except OSError:
                    pass
                    
        except Exception as e:
            error_msg = f"Signature extraction failed: {e}"
            logger.error(f"Page {page_number}: {error_msg}")
            self._save_debug_output(page_number, "signatures", error_msg)
            return []
    
    def process_page(self, image: Image.Image, page_number: int) -> Optional[PageContent]:
        """
        Process a complete page through the full pipeline.
        
        Args:
            image: PIL Image of the page
            page_number: Page number (1-based)
            
        Returns:
            PageContent object with all extracted content, or None if page should be skipped
        """
        analysis = self.analyze_page(image, page_number)
        
        if not analysis.should_parse:
            logger.debug(f"Skipping page {page_number} - no meaningful content")
            return None
        
        page_content = PageContent(page_number=page_number)
        
        # Extract text content
        if analysis.has_text:
            page_content.text_content = self.extract_text_content(image, page_number)
        
        # Extract table content
        if analysis.has_tables:
            page_content.table_content = self.extract_table_content(image, page_number)
            page_content.table_mapping = parse_table_content(page_content.table_content)
        
        # Extract figure content
        if analysis.has_figures:
            page_content.figure_content = self.extract_figure_content(image, page_number)
            page_content.figure_mapping = parse_figure_content(page_content.figure_content)
        
        # Extract signature content - NUEVA FUNCIONALIDAD
        if analysis.has_signatures:
            if (self.config and 
                self.config.task.extract_signatures_in_document_mode and 
                self.signature_analyzer):
                # Extraer datos reales de signatures
                logger.info(f"Extracting signature data from page {page_number}")
                signature_results = self.extract_signature_content(image, page_number)
                page_content.signature_content = signature_results
                page_content.signature_mapping = parse_signature_extraction_results(signature_results)
                
                if signature_results:
                    logger.info(f"Page {page_number}: Found {len(signature_results)} signatures with data extraction")
                else:
                    logger.info(f"Page {page_number}: Signatures detected but no data extracted")
            else:
                # Comportamiento original: solo marcar presencia
                logger.debug(f"Page {page_number}: Signatures detected, using placeholder mode")
                page_content.signature_content = "SIGNATURES_DETECTED"
                page_content.signature_mapping = parse_signature_content("")
        
        # Integrate all content
        if page_content.text_content:
            page_content.integrated_content = integrate_page_content(
                page_content.text_content,
                page_content.table_mapping,
                page_content.figure_mapping,
                page_content.signature_mapping,
                has_tables=analysis.has_tables,
                has_figures=analysis.has_figures,
                has_signatures=analysis.has_signatures,
                extract_signatures=bool(self.config and self.config.task.extract_signatures_in_document_mode)
            )
            
            # Create debug integration output
            debug_info = f"TEXT CONTENT:\n{page_content.text_content}\n\n"
            debug_info += f"TABLE MAPPING:\n{page_content.table_mapping}\n\n"
            debug_info += f"FIGURE MAPPING:\n{page_content.figure_mapping}\n\n"
            debug_info += f"SIGNATURE MAPPING:\n{page_content.signature_mapping}\n\n"
            debug_info += f"INTEGRATED CONTENT:\n{page_content.integrated_content}"
            
            self._save_debug_output(page_number, "integration", debug_info)
        
        return page_content
    
    def get_processing_summary(self, page_content: PageContent) -> str:
        """
        Generate a summary of what was extracted from this page.
        
        Args:
            page_content: PageContent object to summarize
            
        Returns:
            Summary string
        """
        if not page_content:
            return "Page was skipped (no meaningful content)"
        
        summary_parts = []
        
        if page_content.text_content:
            summary_parts.append("text")
        
        if page_content.table_mapping:
            summary_parts.append(f"{len(page_content.table_mapping)} table(s)")
        
        if page_content.figure_mapping:
            summary_parts.append(f"{len(page_content.figure_mapping)} figure(s)")
        
        if page_content.signature_mapping:
            summary_parts.append(f"{len(page_content.signature_mapping)} signature(s)")
        elif page_content.signature_content == "SIGNATURES_DETECTED":
            summary_parts.append("signatures (placeholder mode)")
        
        content_summary = ", ".join(summary_parts) if summary_parts else "no content"
        
        return f"Page {page_content.page_number}: Extracted {content_summary}"
    
    def is_signature_extraction_enabled(self) -> bool:
        """
        Check if signature extraction is enabled for document processing.
        
        Returns:
            True if signature extraction is enabled in document mode
        """
        return (self.config and 
                self.config.task.extract_signatures_in_document_mode and 
                self.signature_analyzer is not None)