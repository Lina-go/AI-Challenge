"""
Unified main entry point for PDF processing and signature extraction.
"""
import argparse
import sys
import logging
from pathlib import Path

from .config import ProcessorConfig, LLMConfig, ProcessingConfig, TaskConfig
from .core.orchestrator import MainOrchestrator, SignatureOrchestrator, DocumentOrchestrator
from .adapters.csv_adapter import CSVSourceAdapter
from .adapters.directory_adapter import DirectorySourceAdapter
from .adapters.url_adapter import URLSourceAdapter
from .utils.logging_config import setup_logging
from .core.document_processor import DocumentProcessor
from .core.llm_interface import create_llm_interface

def main():
    parser = argparse.ArgumentParser(
        description="PDF Processor - Unified document processing and signature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Document processing (extract text, tables, figures)
  python -m pdf_processor_v2.main document single input.pdf --output output.md
  python -m pdf_processor_v2.main document directory /path/to/pdfs/ --output-dir results/
  
  # Signature extraction
  python -m pdf_processor_v2.main signature csv statements.csv --output-dir signature_results/
  python -m pdf_processor_v2.main signature directory /path/to/pdfs/ --preset ollama-moondream
  
  # Using different LLM providers
  python -m pdf_processor_v2.main signature csv data.csv --preset openai
  python -m pdf_processor_v2.main signature csv data.csv --preset anthropic
  python -m pdf_processor_v2.main signature csv data.csv --preset glm-4.5v
  python -m pdf_processor_v2.main signature csv data.csv --preset ollama-llava
  
  # Manual LLM configuration
  python -m pdf_processor_v2.main signature csv data.csv --llm-provider ollama --llm-model "moondream:1.8b"
  python -m pdf_processor_v2.main document single input.pdf --llm-provider huggingface --llm-model "Qwen/Qwen2.5-VL-7B-Instruct"
        """
    )
    
    # Main task selection
    parser.add_argument("task", choices=["document", "signature"],
                       help="Type of processing: 'document' (extract text, tables, figures) or 'signature' (extract signatures)")
    
    # Source configuration
    parser.add_argument("source_type", choices=["single", "csv", "directory", "urls"],
                       help="Source type: single PDF, CSV file, directory, or URL list")
    parser.add_argument("source", help="Path to source (PDF file, CSV file, directory, or URL list)")
    
    # Output configuration
    parser.add_argument("--output", help="Output file path (for single document processing)")
    parser.add_argument("--output-dir", default="results",
                       help="Output directory for results (default: results)")
    
    # Preset configuration
    parser.add_argument("--preset", 
                       choices=["openai", "anthropic", "glm-4.5v", "qwen-vl", 
                               "ollama-moondream", "ollama-llava", "ollama-llava-13b"],
                       help="Use predefined model configuration")
    
    # LLM configuration
    parser.add_argument("--llm-provider", 
                       choices=["openai", "anthropic", "huggingface", "ollama"], 
                       default="openai",
                       help="LLM provider")
    parser.add_argument("--llm-model", default="gpt-4o",
                       help="LLM model to use")
    parser.add_argument("--api-key", help="API key for LLM provider")
    parser.add_argument("--hf-token", help="HuggingFace token (for HuggingFace provider)")
    parser.add_argument("--base-url", help="Base URL for LLM API (for HuggingFace provider)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                       help="Ollama server URL (for Ollama provider)")
    
    # Processing configuration
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of parallel workers")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for PDF to image conversion")
    parser.add_argument("--recursive", action="store_true",
                       help="Recursively scan subdirectories (for directory source)")
    
    # Task-specific options
    parser.add_argument("--extract-tables", action="store_true", default=True,
                       help="Extract tables (document processing)")
    parser.add_argument("--extract-figures", action="store_true", default=True,
                       help="Extract figures (document processing)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate source type compatibility
        if args.source_type == "single" and args.task == "signature":
            logger.error("Single PDF processing not supported for signature extraction. Use 'directory' with single PDF.")
            return 1
        
        # Validate source exists
        source_path = Path(args.source)
        if args.source_type in ["single", "csv", "urls"] and not source_path.exists():
            logger.error(f"Source file not found: {args.source}")
            return 1
        elif args.source_type == "directory" and not source_path.is_dir():
            logger.error(f"Directory not found: {args.source}")
            return 1
        
        # Create configuration
        if args.preset:
            if args.task == "signature":
                config = ProcessorConfig.create_signature_preset(
                    preset=args.preset,
                    output_dir=args.output_dir
                )
            else:
                config = ProcessorConfig.create_preset(
                    preset=args.preset,
                    output_dir=args.output_dir
                )
            
            # Override specific settings if provided
            if args.api_key:
                config.llm.api_key = args.api_key
            if args.hf_token:
                config.llm.hf_token = args.hf_token
            if args.base_url:
                config.llm.base_url = args.base_url
            if args.ollama_url:
                config.llm.ollama_base_url = args.ollama_url
                
        else:
            # Manual configuration
            task_config = TaskConfig(
                task_type="signature_extraction" if args.task == "signature" else "document_processing",
                extract_tables=args.extract_tables if args.task == "document" else False,
                extract_figures=args.extract_figures if args.task == "document" else False,
                extract_signatures=args.task == "signature",
                debug_mode=args.debug
            )
            
            config = ProcessorConfig(
                llm=LLMConfig(
                    provider=args.llm_provider,
                    model=args.llm_model,
                    api_key=args.api_key,
                    hf_token=args.hf_token,
                    base_url=args.base_url,
                    ollama_base_url=args.ollama_url
                ),
                processing=ProcessingConfig(
                    output_dir=args.output_dir,
                    dpi=args.dpi,
                    max_workers=args.max_workers
                ),
                task=task_config
            )
        
        # Add source configuration for recursive scanning
        if args.recursive:
            config.source_config = {'recursive': True}
        
        # Handle single document processing
        if args.source_type == "single":
            
            llm = create_llm_interface(config.llm)
            processor = DocumentProcessor(config)
            
            logger.info(f"Processing single document: {args.source}")
            result = processor.process_document(args.source)
            
            # Save results
            output_path = args.output or f"{source_path.stem}_processed.md"
            processor.save_results(result, output_path)
            
            # Print summary
            summary = processor.get_processing_summary(result)
            print(summary)
            
            logger.info(f"Processing completed. Results saved to: {output_path}")
            return 0
        
        # Create appropriate adapter for multi-source processing
        if args.source_type == "csv":
            adapter = CSVSourceAdapter(config, args.source)
        elif args.source_type == "directory":
            adapter = DirectorySourceAdapter(config, args.source)
        elif args.source_type == "urls":
            adapter = URLSourceAdapter(config, args.source)
        else:
            raise ValueError(f"Unsupported source type: {args.source_type}")
        
        # Create orchestrator
        if args.task == "signature":
            orchestrator = SignatureOrchestrator(config)
        else:
            orchestrator = DocumentOrchestrator(config)
        
        # Get sources
        logger.info(f"Getting sources from {args.source_type}: {args.source}")
        sources = adapter.get_pdf_sources()
        
        if not sources:
            logger.warning("No PDF sources found")
            return 0
        
        # Prepare sources (download if needed)
        logger.info(f"Preparing {len(sources)} sources...")
        for source in sources:
            source['pdf_path'] = adapter.prepare_source(source)
        
        # Process sources
        logger.info(f"Starting {args.task} processing...")
        results = orchestrator.process_multiple_sources(sources)
        
        # Print summary
        total_pdfs = len(results)
        successful = len([r for r in results if not r.get('error')])
        
        if args.task == "signature":
            total_signatures = sum(r.get('signature_count', 0) for r in results)
            print(f"\nSignature Extraction Summary:")
            print(f"PDFs processed: {total_pdfs}")
            print(f"Successful: {successful}")
            print(f"Signatures found: {total_signatures}")
        else:
            print(f"\nDocument Processing Summary:")
            print(f"PDFs processed: {total_pdfs}")
            print(f"Successful: {successful}")
        
        print(f"Results saved to: {args.output_dir}")
        print(f"Model used: {config.llm.provider} - {config.llm.model}")
        
        return 0 if successful == total_pdfs else 1
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())