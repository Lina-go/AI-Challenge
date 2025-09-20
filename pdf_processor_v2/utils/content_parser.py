"""
Content parsing utilities for LLM responses.
"""

import re
from typing import Dict
from ..models.page_analysis import PageAnalysis


def parse_page_analysis(response: str, page_number: int) -> PageAnalysis:
    """
    Parse page analysis response from LLM.
    
    Args:
        response: Raw response from page analysis prompt
        page_number: Page number being analyzed
        
    Returns:
        PageAnalysis object with parsed results
    """
    # Default values
    should_parse = False
    has_text = False
    has_tables = False
    has_figures = False
    has_signatures = False
    
    try:
        # Look for the PAGE_ANALYSIS section
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip().lower()
            
            if 'should_parse:' in line:
                should_parse = 'true' in line
            elif 'has_text:' in line:
                has_text = 'true' in line
            elif 'has_tables:' in line:
                has_tables = 'true' in line
            elif 'has_figures:' in line:
                has_figures = 'true' in line
            elif 'has_signatures:' in line:
                has_signatures = 'true' in line
        
        return PageAnalysis(
            page_number=page_number,
            should_parse=should_parse,
            has_text=has_text,
            has_tables=has_tables,
            has_figures=has_figures,
            has_signatures=has_signatures,
            raw_analysis=response
        )
        
    except Exception as e:
        # Return safe defaults if parsing fails
        return PageAnalysis(
            page_number=page_number,
            should_parse=True,  # Err on side of processing
            has_text=True,
            has_tables=False,
            has_figures=False,
            raw_analysis=f"Parse error: {e}"
        )


def parse_table_content(response: str) -> Dict[str, str]:
    """
    Parse table extraction response into mapping of table IDs to content.
    
    Args:
        response: Raw response from table extraction prompt
        
    Returns:
        Dictionary mapping table IDs (e.g., 'TABLE_1') to markdown content
    """
    table_mapping = {}
    
    try:
        # Split response into sections by TABLE_ identifiers
        # Handle both start-of-string and newline-prefixed cases
        sections = re.split(r'(?:^|\n)(TABLE_\d+)(?:\n|$)', response, flags=re.MULTILINE)
        
        # Process pairs of (table_id, content)
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                table_id = sections[i].strip()
                table_content = sections[i + 1].strip()
                
                if table_content:
                    table_mapping[table_id] = table_content
        
        return table_mapping
        
    except Exception:
        # Return empty mapping if parsing fails
        return {}


def parse_figure_content(response: str) -> Dict[str, str]:
    """
    Parse figure extraction response into mapping of figure IDs to content.
    
    Args:
        response: Raw response from figure extraction prompt
        
    Returns:
        Dictionary mapping figure IDs (e.g., 'FIGURE_1') to markdown content
    """
    figure_mapping = {}
    
    try:
        # Split response into sections by FIGURE_ identifiers
        # Handle both start-of-string and newline-prefixed cases
        sections = re.split(r'(?:^|\n)(FIGURE_\d+)(?:\n|$)', response, flags=re.MULTILINE)
        
        # Process pairs of (figure_id, content)
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                figure_id = sections[i].strip()
                figure_content = sections[i + 1].strip()
                
                if figure_content:
                    figure_mapping[figure_id] = figure_content
        
        return figure_mapping
        
    except Exception:
        # Return empty mapping if parsing fails
        return {}


def parse_signature_content(response: str) -> Dict[str, str]:
    """
    Parse signature content response from LLM.
    
    Args:
        response: Raw response from signature extraction prompt
        
    Returns:
        Dictionary mapping signature IDs to content (empty since signatures remain as placeholders)
    """
    # For signatures, we don't actually parse content since they remain as placeholders
    # This function exists for consistency but returns empty mapping
    return {}


def extract_placeholders(text: str) -> Dict[str, list[str]]:
    """
    Extract all placeholders from text content.
    
    Args:
        text: Text content with placeholders
        
    Returns:
        Dictionary with 'tables', 'figures', and 'signatures' keys containing lists of placeholder IDs
    """
    if not text:
        return {"tables": [], "figures": [], "signatures": []}
    
    table_placeholders = re.findall(r'\[TABLE_(\d+)\]', text)
    figure_placeholders = re.findall(r'\[FIGURE_(\d+)\]', text)
    signature_placeholders = re.findall(r'\[SIGNATURE_(\d+)\]', text)
    
    return {
        "tables": [f"TABLE_{num}" for num in table_placeholders],
        "figures": [f"FIGURE_{num}" for num in figure_placeholders],
        "signatures": [f"SIGNATURE_{num}" for num in signature_placeholders]
    }
