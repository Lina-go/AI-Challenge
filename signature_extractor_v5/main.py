# signature_extractor_v5/main.py
"""Command-line interface."""

import argparse
import logging
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

from .config import ExtractionConfig, DoclingConfig, ProcessingConfig
from .core import ExtractionOrchestrator
from .adapters import CSVSourceAdapter, DirectorySourceAdapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Signature Extractor v5 - Docling VLM-powered extraction"
    )
    
    parser.add_argument(
        "source_type",
        choices=["csv", "directory"],
        help="Source type"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to source"
    )
    parser.add_argument(
        "--output-dir",
        default="signature_results",
        help="Output directory"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Maximum pages per document"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model (auto/cpu/cuda)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = ExtractionConfig(
        docling=DoclingConfig(device=args.device),
        processing=ProcessingConfig(
            output_dir=args.output_dir,
            max_pages=args.max_pages
        )
    )
    
    # Create adapter
    if args.source_type == "csv":
        adapter = CSVSourceAdapter(
            args.source,
            download_dir=f"{args.output_dir}/downloads"
        )
    else:
        adapter = DirectorySourceAdapter(args.source)
    
    # Get sources
    sources = adapter.get_sources()
    
    # Prepare sources
    for source in sources:
        adapter.prepare_source(source)
    
    # Process documents
    orchestrator = ExtractionOrchestrator(config)
    results = orchestrator.process_batch(sources)
    
    # Save results
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(args.output_dir) / f"signatures_{timestamp}.csv"
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Results saved to: {output_file}")
        print(f"\nExtraction complete:")
        print(f"  Documents processed: {len(sources)}")
        print(f"  Signatures found: {len(results)}")
        print(f"  Output: {output_file}")
    else:
        logger.warning("No results to save")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())