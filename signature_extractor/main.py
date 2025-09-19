import argparse
import sys
from pathlib import Path
from .adapters.csv_adapter import CSVSourceAdapter
from .core.orchestrator import ExtractionOrchestrator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="PDF Signature Extractor")
    
    parser.add_argument(
        "--source-type", 
        choices=["csv", "directory", "url"],
        required=True,
        help="Type of PDF source"
    )
    
    parser.add_argument(
        "--source",
        required=True, 
        help="Path to CSV file, directory, or PDF URL"
    )
    
    parser.add_argument(
        "--output-dir",
        default="signature_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for signature detection"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = ExtractorConfig(
        detection=DetectionConfig(conf_threshold=args.conf_threshold),
        processing=ProcessingConfig(output_dir=args.output_dir)
    )
    
    # Select appropriate adapter
    if args.source_type == "csv":
        adapter = CSVSourceAdapter(config, args.source)
    # Add other adapters as needed
    else:
        raise ValueError(f"Unsupported source type: {args.source_type}")
    
    # Run extraction
    orchestrator = ExtractionOrchestrator(config)
    sources = adapter.get_pdf_sources()
    
    results = []
    for source in sources:
        pdf_path = adapter.prepare_source(source)
        result = orchestrator.process_pdf(pdf_path, source['source_id'])
        results.append(result)
    
    print(f"Extraction completed: {len(results)} PDFs processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())