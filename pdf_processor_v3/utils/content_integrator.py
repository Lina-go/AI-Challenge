"""
Content integration utilities for combining text with tables and figures.
"""

import re
from typing import Dict


def integrate_page_content(
    text_content: str,
    table_mapping: Dict[str, str],
    figure_mapping: Dict[str, str],
    signature_mapping: Dict[str, str] = None,
    has_tables: bool = True,
    has_figures: bool = True,
    has_signatures: bool = False,
    extract_signatures: bool = False
) -> str:
    """
    Integrate text content with tables, figures, and optionally signatures.
    
    Args:
        text_content: Text with placeholders like [TABLE_1], [FIGURE_2], [SIGNATURE_1]
        table_mapping: Mapping of table IDs to markdown content
        figure_mapping: Mapping of figure IDs to markdown content
        signature_mapping: Mapping of signature IDs to markdown content
        has_tables: Whether the page actually has tables (from analysis)
        has_figures: Whether the page actually has figures (from analysis)
        has_signatures: Whether the page actually has signatures (from analysis)
        extract_signatures: Whether to extract signature content or leave as placeholders
        
    Returns:
        Integrated markdown content with placeholders replaced
    """
    if not text_content:
        return ""
    
    integrated = text_content
    
    # If analysis says no tables exist, remove any hallucinated table placeholders
    if not has_tables:
        integrated = re.sub(r'\[TABLE_\d+\]', '', integrated)
    
    # If analysis says no figures exist, remove any hallucinated figure placeholders  
    if not has_figures:
        integrated = re.sub(r'\[FIGURE_\d+\]', '', integrated)
    
    # If analysis says no signatures exist, remove any hallucinated signature placeholders
    if not has_signatures:
        integrated = re.sub(r'\[SIGNATURE_\d+\]', '', integrated)
    
    # Replace table placeholders with headers and content
    for table_id, table_content in table_mapping.items():
        placeholder = f"[{table_id}]"
        if placeholder in integrated:
            if table_content.strip() == "NOT_INFORMATIVE":
                # Leave placeholder with explanation for non-informative tables
                replacement = f"\n\n[{table_id}] - NOT_INFORMATIVE table (not parsed)\n\n"
            else:
                # Extract table number for header
                table_num = table_id.split('_')[1]
                formatted_table = f"\n\n## Table {table_num}\n\n{table_content}\n\n"
                replacement = formatted_table
            integrated = integrated.replace(placeholder, replacement)
    
    # Replace figure placeholders with headers and content
    for figure_id, figure_content in figure_mapping.items():
        placeholder = f"[{figure_id}]"
        if placeholder in integrated:
            if figure_content.strip() == "NOT_INFORMATIVE":
                # Leave placeholder with explanation for non-informative figures
                replacement = f"\n\n[{figure_id}] - NOT_INFORMATIVE figure (not parsed)\n\n"
            else:
                # Extract figure number for header
                figure_num = figure_id.split('_')[1]
                formatted_figure = f"\n\n## Figure {figure_num}\n\n{figure_content}\n\n"
                replacement = formatted_figure
            integrated = integrated.replace(placeholder, replacement)
    
    # NUEVA FUNCIONALIDAD: Replace signature placeholders with content if extraction is enabled
    if extract_signatures and signature_mapping:
        for signature_id, signature_content in signature_mapping.items():
            placeholder = f"[{signature_id}]"
            if placeholder in integrated:
                if signature_content.strip():
                    # Extract signature number for header
                    signature_num = signature_id.split('_')[1]
                    formatted_signature = f"\n\n## Signature {signature_num}\n\n{signature_content}\n\n"
                    replacement = formatted_signature
                else:
                    # Leave placeholder if no content
                    replacement = f"\n\n[{signature_id}] - No signature data extracted\n\n"
                integrated = integrated.replace(placeholder, replacement)
    elif has_signatures and not extract_signatures:
        # Signatures remain as placeholders (original behavior)
        pass
    
    # Clean up extra whitespace
    integrated = re.sub(r'\n{3,}', '\n\n', integrated)
    
    return integrated.strip()


def validate_integration(
    text_content: str,
    table_mapping: Dict[str, str],
    figure_mapping: Dict[str, str],
    integrated_content: str
) -> Dict[str, any]:
    """
    Validate that integration was successful by checking for remaining placeholders.
    
    Args:
        text_content: Original text content
        table_mapping: Table mapping used for integration
        figure_mapping: Figure mapping used for integration
        integrated_content: Final integrated content
        
    Returns:
        Dictionary with validation results
    """
    # Find remaining placeholders
    remaining_table_placeholders = re.findall(r'\[TABLE_\d+\]', integrated_content)
    remaining_figure_placeholders = re.findall(r'\[FIGURE_\d+\]', integrated_content)
    
    # Find expected placeholders from original text
    expected_table_placeholders = re.findall(r'\[TABLE_\d+\]', text_content)
    expected_figure_placeholders = re.findall(r'\[FIGURE_\d+\]', text_content)
    
    # Calculate success rates
    total_expected = len(expected_table_placeholders) + len(expected_figure_placeholders)
    total_remaining = len(remaining_table_placeholders) + len(remaining_figure_placeholders)
    
    success_rate = 0.0 if total_expected == 0 else ((total_expected - total_remaining) / total_expected) * 100
    
    return {
        "success_rate": success_rate,
        "total_placeholders": total_expected,
        "remaining_placeholders": total_remaining,
        "remaining_table_placeholders": remaining_table_placeholders,
        "remaining_figure_placeholders": remaining_figure_placeholders,
        "missing_tables": [p for p in expected_table_placeholders if p in remaining_table_placeholders],
        "missing_figures": [p for p in expected_figure_placeholders if p in remaining_figure_placeholders],
        "available_tables": list(table_mapping.keys()),
        "available_figures": list(figure_mapping.keys())
    }


def combine_page_contents(page_contents) -> str:
    """
    Combine multiple page contents into a single document.
    
    Args:
        page_contents: List of PageContent objects
        
    Returns:
        Combined markdown document
    """
    if not page_contents:
        return ""
    
    # Extract integrated content from PageContent objects
    integrated_contents = []
    for page_content in page_contents:
        if hasattr(page_content, 'integrated_content') and page_content.integrated_content:
            integrated_contents.append(page_content.integrated_content)
        elif hasattr(page_content, 'text_content') and page_content.text_content:
            # Fallback to raw text content if integration failed
            integrated_contents.append(page_content.text_content)
    
    if not integrated_contents:
        return ""
    
    # Join with page breaks
    combined = "\n\n---\n\n".join(integrated_contents)
    
    # Clean up excessive whitespace
    combined = re.sub(r'\n{3,}', '\n\n', combined)
    
    return combined.strip()