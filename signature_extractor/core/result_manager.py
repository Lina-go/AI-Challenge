import polars as pl
import json
import os
from datetime import datetime
from typing import List, Dict, Any


class ResultManager:
    """
    Manages result collection, formatting, and saving.
    
    Single Responsibility: Result persistence and reporting.
    """
    
    def __init__(self, config):
        self.config = config
        self.all_results = []
    
    def save_pdf_results(
        self, 
        source_id: str, 
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Save results for a single PDF.
        
        Args:
            source_id: Unique identifier for the PDF
            detections: List of detection results
            
        Returns:
            Summary dictionary for this PDF
        """
        # Create result summary
        summary = {
            'source_id': source_id,
            'timestamp': datetime.now().isoformat(),
            'signatures_found': len(detections),
            'pages_processed': len(set(d['page_number'] for d in detections)),
            'detections': detections
        }
        
        # Save individual PDF results
        self._save_individual_results(source_id, summary)
        
        # Add to overall collection
        self.all_results.append(summary)
        
        return summary
    
    def save_final_results(self) -> str:
        """
        Save final combined results.
        
        Returns:
            Path to saved results file
        """
        output_file = os.path.join(
            self.config.processing.output_dir,
            "signature_extraction_results.json"
        )
        
        final_results = {
            'extraction_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_pdfs': len(self.all_results),
                'total_signatures': sum(r['signatures_found'] for r in self.all_results),
                'configuration': self._config_to_dict()
            },
            'pdf_results': self.all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Final results saved to: {output_file}")
        return output_file
    
    def create_summary_dataframe(self) -> pl.DataFrame:
        """Create a summary DataFrame of all results."""
        summary_rows = []
        
        for result in self.all_results:
            summary_rows.append({
                'source_id': result['source_id'],
                'signatures_found': result['signatures_found'],
                'pages_processed': result['pages_processed'],
                'timestamp': result['timestamp']
            })
        
        return pl.DataFrame(summary_rows)
    
    def _save_individual_results(self, source_id: str, summary: Dict[str, Any]):
        """Save results for individual PDF."""
        result_file = os.path.join(
            self.config.processing.output_dir,
            f"{source_id}_results.json"
        )
        
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'detection': {
                'model_name': self.config.detection.model_name,
                'conf_threshold': self.config.detection.conf_threshold,
                'ensure_one_per_page': self.config.detection.ensure_one_per_page,
                'device': self.config.detection.device
            },
            'processing': {
                'output_dir': self.config.processing.output_dir,
                'save_crops': self.config.processing.save_crops,
                'save_page_images': self.config.processing.save_page_images,
                'max_workers': self.config.processing.max_workers
            }
        }