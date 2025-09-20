import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ResultManager:
    """Manages saving and organizing extraction results"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.processing.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.csv_dir = self.output_dir / "csv_results"
        self.json_dir = self.output_dir / "json_results"
        self.individual_dir = self.output_dir / "individual_results"
        
        for dir_path in [self.csv_dir, self.json_dir, self.individual_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_source_results(self, source_id: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Save results for a single source
        
        Args:
            source_id: Unique identifier for the source
            results: List of signature extraction results
            
        Returns:
            Summary of saved results
        """
        try:
            # Save individual JSON result
            json_file = self.individual_dir / f"{source_id}_results.json"
            result_data = {
                'source_id': source_id,
                'extraction_timestamp': datetime.now().isoformat(),
                'signature_count': len(results),
                'signatures': results,
                'summary': self._create_source_summary(results)
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            # Save individual CSV result
            csv_file = self.individual_dir / f"{source_id}_results.csv"
            if results:
                df = pd.DataFrame(results)
                df.to_csv(csv_file, index=False)
            
            logger.info(f"Saved results for {source_id}: {len(results)} signatures")
            
            return {
                'source_id': source_id,
                'signature_count': len(results),
                'has_signatures': any(r.get('Signature') == 'Yes' for r in results),
                'json_file': str(json_file),
                'csv_file': str(csv_file) if results else None,
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error saving results for {source_id}: {e}")
            return {
                'source_id': source_id,
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def save_consolidated_results(self, all_results: List[Dict[str, Any]]):
        """
        Save consolidated results from all sources
        
        Args:
            all_results: List of all extraction results
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Flatten all signature results
            all_signatures = []
            processing_summary = []
            
            for source_result in all_results:
                if 'error' not in source_result:
                    # This should contain the processed signature data
                    signatures = source_result.get('signatures', [])
                    all_signatures.extend(signatures)
                
                # Add to processing summary
                processing_summary.append({
                    'source_id': source_result.get('source_id', ''),
                    'signature_count': source_result.get('signature_count', 0),
                    'has_signatures': source_result.get('has_signatures', False),
                    'processing_status': 'success' if 'error' not in source_result else 'error',
                    'error_message': source_result.get('error', ''),
                    'processing_timestamp': source_result.get('processing_timestamp', '')
                })
            
            # Save consolidated CSV
            csv_file = self.csv_dir / f"signature_extraction_results_{timestamp}.csv"
            if all_signatures:
                df = pd.DataFrame(all_signatures)
                
                # Ensure we have the required columns in the right order
                required_columns = [
                    'Signature', 'Signature_Yes_text', 'Signature_Image',
                    'Signature_scanned', 'Presence_Signature_date', 'Signature_Date'
                ]
                
                # Add missing columns if needed
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = ''
                
                # Reorder columns
                other_columns = [col for col in df.columns if col not in required_columns]
                final_columns = required_columns + other_columns
                df = df[final_columns]
                
                df.to_csv(csv_file, index=False)
                logger.info(f"Saved consolidated CSV: {csv_file}")
            
            # Save consolidated JSON
            json_file = self.json_dir / f"signature_extraction_results_{timestamp}.json"
            consolidated_data = {
                'extraction_summary': {
                    'total_sources': len(all_results),
                    'successful_sources': len([r for r in all_results if 'error' not in r]),
                    'total_signatures': len(all_signatures),
                    'sources_with_signatures': len([r for r in processing_summary if r['has_signatures']]),
                    'extraction_timestamp': datetime.now().isoformat()
                },
                'processing_summary': processing_summary,
                'signature_results': all_signatures
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved consolidated JSON: {json_file}")
            
            # Save processing summary CSV
            summary_csv = self.output_dir / f"processing_summary_{timestamp}.csv"
            summary_df = pd.DataFrame(processing_summary)
            summary_df.to_csv(summary_csv, index=False)
            
            # Create final summary report
            self._create_summary_report(consolidated_data, timestamp)
            
        except Exception as e:
            logger.error(f"Error saving consolidated results: {e}")
            raise
    
    def _create_source_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics for a single source"""
        if not results:
            return {
                'total_signatures': 0,
                'image_signatures': 0,
                'scanned_signatures': 0,
                'signatures_with_dates': 0
            }
        
        return {
            'total_signatures': len(results),
            'image_signatures': len([r for r in results if r.get('Signature_Image') == 'Yes']),
            'scanned_signatures': len([r for r in results if r.get('Signature_scanned') == 'Yes']),
            'signatures_with_dates': len([r for r in results if r.get('Presence_Signature_date') == 'Yes']),
            'pages_with_signatures': len(set(r.get('page_number', 0) for r in results if r.get('Signature') == 'Yes'))
        }
    
    def _create_summary_report(self, consolidated_data: Dict[str, Any], timestamp: str):
        """Create a human-readable summary report"""
        try:
            report_file = self.output_dir / f"extraction_report_{timestamp}.txt"
            
            summary = consolidated_data['extraction_summary']
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("SIGNATURE EXTRACTION REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Extraction completed: {summary['extraction_timestamp']}\n\n")
                
                f.write("OVERVIEW:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total sources processed: {summary['total_sources']}\n")
                f.write(f"Successful extractions: {summary['successful_sources']}\n")
                f.write(f"Total signatures found: {summary['total_signatures']}\n")
                f.write(f"Sources with signatures: {summary['sources_with_signatures']}\n\n")
                
                # Add detailed breakdown
                processing_summary = consolidated_data['processing_summary']
                successful = [p for p in processing_summary if p['processing_status'] == 'success']
                failed = [p for p in processing_summary if p['processing_status'] == 'error']
                
                if successful:
                    f.write("SUCCESSFUL EXTRACTIONS:\n")
                    f.write("-" * 30 + "\n")
                    for proc in successful:
                        f.write(f"- {proc['source_id']}: {proc['signature_count']} signatures\n")
                    f.write("\n")
                
                if failed:
                    f.write("FAILED EXTRACTIONS:\n")
                    f.write("-" * 25 + "\n")
                    for proc in failed:
                        f.write(f"- {proc['source_id']}: {proc['error_message']}\n")
                    f.write("\n")
                
                f.write("OUTPUT FILES:\n")
                f.write("-" * 15 + "\n")
                f.write(f"- Consolidated CSV: csv_results/signature_extraction_results_{timestamp}.csv\n")
                f.write(f"- Consolidated JSON: json_results/signature_extraction_results_{timestamp}.json\n")
                f.write(f"- Processing Summary: processing_summary_{timestamp}.csv\n")
                f.write(f"- Individual Results: individual_results/\n")
            
            logger.info(f"Created summary report: {report_file}")
            
        except Exception as e:
            logger.error(f"Error creating summary report: {e}")