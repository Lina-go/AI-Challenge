"""
Main entry point for PDF processing pipeline.
"""

import logging
import argparse
from pathlib import Path

from .core.document_processor import DocumentProcessor
from .core.llm_factory import create_llm_interface, get_available_providers, get_recommended_provider


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process PDF documents to extract structured content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF with auto-selected LLM
  python -m pdf_processor.main input.pdf --output output.md
  
  # Process with GPT-4 Vision
  python -m pdf_processor.main input.pdf --output output.md --provider gpt
  
  # Process with Gemma on CPU
  python -m pdf_processor.main input.pdf --output output.md --provider gemma --device cpu
  
  # Process with debug logging
  python -m pdf_processor.main input.pdf --output output.md --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to process"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="output.md",
        help="Output file path for extracted content (default: output.md)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--provider", "-p",
        choices=["gpt", "gemma", "auto"],
        default="auto",
        help="LLM provider: gpt, gemma, or auto-detect (default: auto)"
    )
    
    parser.add_argument(
        "--device", "-d",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for local models (Gemma): cpu, cuda, or auto-detect (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input file
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return 1
        
        if not pdf_path.suffix.lower() == '.pdf':
            logger.error(f"Input file must be a PDF: {pdf_path}")
            return 1
        
        # Determine LLM provider
        if args.provider == "auto":
            provider = get_recommended_provider()
            logger.info(f"Auto-selected LLM provider: {provider}")
        else:
            provider = args.provider
        
        # Show available providers
        available = get_available_providers()
        logger.info(f"Available LLM providers: {available}")
        
        # Initialize LLM interface
        logger.info(f"Initializing PDF processor with {provider.upper()} LLM interface")
        
        if provider == "gpt":
            llm = create_llm_interface("gpt")
        elif provider == "gemma":
            llm = create_llm_interface("gemma", device=args.device)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"LLM interface ready: {llm.get_provider_name()}")
        processor = DocumentProcessor(llm)
        
        # Process document
        logger.info(f"Processing PDF: {pdf_path}")
        result = processor.process_document(str(pdf_path))
        
        # Save results
        output_path = Path(args.output)
        processor.save_results(result, str(output_path))
        
        # Print summary
        summary = processor.get_processing_summary(result)
        print(summary)
        
        logger.info("Processing completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
