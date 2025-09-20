# main.py
import argparse
import sys
from pathlib import Path
from .config import ExtractionConfig, LLMConfig, ProcessingConfig
from .core.orchestrator import ExtractionOrchestrator
from .adapters.csv_adapter import CSVSourceAdapter
from .adapters.directory_adapter import DirectorySourceAdapter
from .adapters.url_adapter import URLSourceAdapter
from .utils.logging_config import setup_logging

def main():
    parser = argparse.ArgumentParser(
        description="PDF Signature Extractor v2 - LLM-powered signature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from CSV file with PDF URLs
  python -m signature_extractor_v2.main csv --source statements.csv
  
  # Extract from directory of PDFs
  python -m signature_extractor_v2.main directory --source /path/to/pdfs/
  
  # Extract from URL list with custom config
  python -m signature_extractor_v2.main urls --source urls.txt --output-dir results/ --llm-model gpt-4o
  
  # Use specific LLM provider
  python -m signature_extractor_v2.main csv --source data.csv --llm-provider anthropic --llm-model claude-3-sonnet
        """
    )
    
    # Source configuration
    parser.add_argument("source_type", choices=["csv", "directory", "urls"],
                       help="Type of PDF source")
    parser.add_argument("--source", required=True,
                       help="Path to source (CSV file, directory, or URL list)")
    
    # Output configuration
    parser.add_argument("--output-dir", default="signature_results",
                       help="Output directory for results")
    parser.add_argument("--format", choices=["csv", "json", "both"], default="csv",
                       help="Output format")
    
    # LLM configuration
    parser.add_argument("--llm-provider", choices=["openai", "anthropic"], default="openai",
                       help="LLM provider")
    parser.add_argument("--llm-model", default="gpt-4o",
                       help="LLM model to use")
    parser.add_argument("--api-key", help="API key for LLM provider")
    
    # Processing configuration
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of parallel workers")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for PDF to image conversion")
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create configuration
    config = ExtractionConfig(
        llm=LLMConfig(
            provider=args.llm_provider,
            model=args.llm_model,
            api_key=args.api_key
        ),
        processing=ProcessingConfig(
            output_dir=args.output_dir,
            dpi=args.dpi,
            max_workers=args.max_workers
        )
    )
    
    # Create appropriate adapter
    if args.source_type == "csv":
        adapter = CSVSourceAdapter(config, args.source)
    elif args.source_type == "directory":
        adapter = DirectorySourceAdapter(config, args.source)
    else:  # urls
        adapter = URLSourceAdapter(config, args.source)
    
    # Process sources
    orchestrator = ExtractionOrchestrator(config)
    sources = adapter.get_pdf_sources()
    
    # Prepare sources (download if needed)
    for source in sources:
        source['pdf_path'] = adapter.prepare_source(source)
    
    # Extract signatures
    results = orchestrator.process_multiple_sources(sources)
    
    # Print summary
    total_pdfs = len(results)
    successful = len([r for r in results if not r.get('error')])
    total_signatures = sum(r.get('signature_count', 0) for r in results)
    
    print(f"\nExtraction Summary:")
    print(f"PDFs processed: {total_pdfs}")
    print(f"Successful: {successful}")
    print(f"Signatures found: {total_signatures}")
    print(f"Results saved to: {args.output_dir}")
    
    return 0 if successful == total_pdfs else 1

if __name__ == "__main__":
    sys.exit(main())