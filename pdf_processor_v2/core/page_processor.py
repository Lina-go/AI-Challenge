"""
Page-level processing logic for PDF pipeline with integrated signature extraction.
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
from ..utils.content_parser import parse_page_analysis, parse_table_content, parse_figure_content, parse_signature_content
from ..utils.content_integrator import integrate_page_content

logger = logging.getLogger(__name__)


class PageProcessor:
    """Handles processing of individual PDF pages with integrated signature extraction."""
    
    def __init__(self, llm_interface: LLMInterface, config=None, debug_dir: Optional[str] = None):
        """
        Initialize page processor.
        
        Args:
            llm_interface: LLM interface for processing images
            config: ProcessorConfig object (for signature extraction)
            debug_dir: Optional directory to save debug outputs
        """
        self.llm = llm_interface
        self.config = config
        self.debug_dir = debug_dir or "pdf_processor/debug_logs"
        
        # Initialize signature analyzer if config provided
        if config:
            self.signature_analyzer = SignatureAnalyzer(config)
        else:
            self.signature_analyzer = None
        
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
    
    def extract_signature_data(self, image: Image.Image, page_number: int) -> dict:
        """
        Extract signature data from a page using SignatureAnalyzer.
        
        Args:
            image: PIL Image of the page
            page_number: Page number for logging
            
        Returns:
            Dictionary with signature data for table generation
        """
        if not self.signature_analyzer:
            logger.warning("No signature analyzer available")
            return {}
        
        try:
            # Save image temporarily for signature analyzer
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image.save(tmp_file.name, 'PNG')
                temp_image_path = tmp_file.name
            
            try:
                # Use existing signature analyzer
                signature_results = self.signature_analyzer.analyze_page(temp_image_path, page_number)
                
                # Format for table generation
                signatures_data = []
                for i, sig_result in enumerate(signature_results, 1):
                    signature_data = {
                        'page': page_number,
                        'name': self._extract_name_from_signatory_text(sig_result.get('signatory_text', '')),
                        'role': self._extract_role_from_signatory_text(sig_result.get('signatory_text', '')),
                        'source': 'document_processing',
                        'raw_text_len': len(sig_result.get('signatory_text', '')),
                        'context': sig_result.get('signatory_text', '').replace('\n', ' ')[:100] + '...' if len(sig_result.get('signatory_text', '')) > 100 else sig_result.get('signatory_text', '').replace('\n', ' ')
                    }
                    signatures_data.append(signature_data)
                
                self._save_debug_output(page_number, "signatures", str(signature_results))
                
                return {'signatures': signatures_data}
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_image_path)
                except OSError:
                    pass
                    
        except Exception as e:
            logger.error(f"Signature extraction failed for page {page_number}: {e}")
            return {}
    
    def _extract_name_from_signatory_text(self, text: str) -> str:
        """Extract name from signatory text."""
        if not text:
            return ""
        
        lines = text.strip().split('\n')
        # Usually the first non-empty line is the name
        for line in lines:
            line = line.strip()
            if line and not any(keyword in line.lower() for keyword in ['director', 'ceo', 'officer', 'manager', 'date', 'company']):
                return line
        
        # Fallback to first line if no clear name found
        return lines[0].strip() if lines else ""
    
    def _extract_role_from_signatory_text(self, text: str) -> str:
        """Extract role/title from signatory text."""
        if not text:
            return ""
        
        lines = text.strip().split('\n')
        role_keywords = ['director', 'ceo', 'chief', 'officer', 'manager', 'president', 'secretary']
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in role_keywords):
                return line.strip()
        
        return ""
    
    def _format_signatures_table(self, signatures_data: list) -> str:
        """Format signature data as markdown table."""
        if not signatures_data:
            return ""
        
        table = "| Page | Name | Role | Source | Raw Text Length | Context |\n"
        table += "|------|------|------|--------|-----------------|----------|\n"
        
        for sig in signatures_data:
            table += f"| {sig['page']} | {sig['name']} | {sig['role']} | {sig['source']} | {sig['raw_text_len']} | {sig['context']} |\n"
        
        return table
    
    def process_page(self, image: Image.Image, page_number: int) -> Optional[PageContent]:
        """
        Process a complete page through the full pipeline with signature extraction.
        
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
        
        # NEW: Extract signature data when signatures detected
        if analysis.has_signatures:
            signature_data = self.extract_signature_data(image, page_number)
            
            if signature_data and signature_data.get('signatures'):
                # Format as markdown table
                signatures_table = self._format_signatures_table(signature_data['signatures'])
                
                # Store formatted table in signature_mapping
                page_content.signature_mapping = {'SIGNATURE_1': signatures_table}
                page_content.signature_content = "SIGNATURES_DETECTED_AND_EXTRACTED"
                
                logger.info(f"Extracted {len(signature_data['signatures'])} signatures from page {page_number}")
            else:
                # Keep original behavior if extraction fails
                page_content.signature_content = "SIGNATURES_DETECTED"
                page_content.signature_mapping = parse_signature_content("")
        
        if page_content.text_content:
            page_content.integrated_content = integrate_page_content(
                page_content.text_content,
                page_content.table_mapping,
                page_content.figure_mapping,
                page_content.signature_mapping,
                has_tables=analysis.has_tables,
                has_figures=analysis.has_figures,
                has_signatures=analysis.has_signatures  # Pass this info
            )
            
            debug_info = f"TEXT CONTENT:\n{page_content.text_content}\n\n"
            debug_info += f"TABLE MAPPING:\n{page_content.table_mapping}\n\n"
            debug_info += f"FIGURE MAPPING:\n{page_content.figure_mapping}\n\n"
            debug_info += f"SIGNATURE MAPPING:\n{page_content.signature_mapping}\n\n"
            debug_info += f"INTEGRATED CONTENT:\n{page_content.integrated_content}"
            
            self._save_debug_output(page_number, "integration", debug_info)
        
        return page_content