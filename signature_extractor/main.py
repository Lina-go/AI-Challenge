import argparse
import sys
from .core.orchestrator import ExtractionOrchestrator
from .config import ExtractorConfig, DetectionConfig, ProcessingConfig
from .adapters.csv_adapter import CSVSourceAdapter
from .adapters.directory_adapter import DirectorySourceAdapter
from .utils.logging import setup_logging


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="PDF Signature Extractor")
    
    parser.add_argument("--source-type", choices=["csv", "directory"], required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", default="signature_results")
    parser.add_argument("--conf-threshold", type=float, default=0.6)
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Create configuration
    config = ExtractorConfig(
        detection=DetectionConfig(conf_threshold=args.conf_threshold),
        processing=ProcessingConfig(output_dir=args.output_dir)
    )
    
    # Select adapter
    if args.source_type == "csv":
        adapter = CSVSourceAdapter(config, args.source)
    else:  # directory
        adapter = DirectorySourceAdapter(config, args.source)
    
    # Process PDFs
    orchestrator = ExtractionOrchestrator(config)
    sources = adapter.get_pdf_sources()
    
    # Prepare and add pdf_path to sources
    for source in sources:
        source['pdf_path'] = adapter.prepare_source(source)
    
    results = orchestrator.process_multiple_pdfs(sources)
    
    # Summary
    successful = len([r for r in results if 'error' not in r])
    total_signatures = sum(r.get('signatures_found', 0) for r in results)
    
    print(f"Processed {len(results)} PDFs")
    print(f"Successful: {successful}")
    print(f"Signatures found: {total_signatures}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())