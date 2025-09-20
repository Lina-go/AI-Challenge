"""
Page-level processing logic for PDF pipeline.
"""

import logging
from pathlib import Path
from typing import Optional
from PIL import Image

from .llm_interface import LLMInterface
from ..models.page_analysis import PageAnalysis
from ..models.page_content import PageContent
from ..utils.prompt_loader import load_prompt
from ..utils.content_parser import parse_page_analysis, parse_table_content, parse_figure_content, parse_signature_content
from ..utils.content_integrator import integrate_page_content

logger = logging.getLogger(__name__)


class PageProcessor:
    """Handles processing of individual PDF pages."""
    
    def __init__(self, llm_interface: LLMInterface, debug_dir: Optional[str] = None):
        """
        Initialize page processor.
        
        Args:
            llm_interface: LLM interface for processing images
            debug_dir: Optional directory to save debug outputs
        """
        self.llm = llm_interface
        self.debug_dir = debug_dir or "pdf_processor_v2/debug_logs"
        
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
            return None
        
        page_content = PageContent(page_number=page_number)
        
        if analysis.has_text:
            page_content.text_content = self.extract_text_content(image, page_number)
        
        if analysis.has_tables:
            page_content.table_content = self.extract_table_content(image, page_number)
            page_content.table_mapping = parse_table_content(page_content.table_content)
        
        if analysis.has_figures:
            page_content.figure_content = self.extract_figure_content(image, page_number)
            page_content.figure_mapping = parse_figure_content(page_content.figure_content)
        
        if analysis.has_signatures:
            # Note: We don't actually extract signature content since they remain as placeholders
            # This is just for consistency in tracking
            page_content.signature_content = "SIGNATURES_DETECTED"
            page_content.signature_mapping = parse_signature_content("")
        
        if page_content.text_content:
            page_content.integrated_content = integrate_page_content(
                page_content.text_content,
                page_content.table_mapping,
                page_content.figure_mapping,
                page_content.signature_mapping,
                has_tables=analysis.has_tables,
                has_figures=analysis.has_figures
            )
            
            debug_info = f"TEXT CONTENT:\n{page_content.text_content}\n\n"
            debug_info += f"TABLE MAPPING:\n{page_content.table_mapping}\n\n"
            debug_info += f"FIGURE MAPPING:\n{page_content.figure_mapping}\n\n"
            debug_info += f"SIGNATURE MAPPING:\n{page_content.signature_mapping}\n\n"
            debug_info += f"INTEGRATED CONTENT:\n{page_content.integrated_content}"
            
            self._save_debug_output(page_number, "integration", debug_info)
        
        return page_content