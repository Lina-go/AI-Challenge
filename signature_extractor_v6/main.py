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
        description="PDF Signature Extractor v3 - LLM-powered signature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from CSV file with OpenAI GPT-4o
  python -m signature_extractor_v3.main csv --source statements.csv
  
  # Extract using HuggingFace GLM-4.5V model
  python -m signature_extractor_v3.main csv --source statements.csv --preset glm-4.5v
  
  # Extract using Ollama Moondream model
  python -m signature_extractor_v3.main csv --source statements.csv --preset ollama-moondream
  
  # Extract from directory using Ollama preset
  python -m signature_extractor_v3.main directory --source /path/to/pdfs/ --preset ollama-llava
  
  # Manual Ollama configuration
  python -m signature_extractor_v3.main csv --source statements.csv --llm-provider ollama --llm-model "moondream:1.8b"
  
  # Manual HuggingFace configuration
  python -m signature_extractor_v3.main csv --source statements.csv --llm-provider huggingface --llm-model "zai-org/GLM-4.5V:novita"
  
  # Extract from URL list with custom config
  python -m signature_extractor_v3.main urls --source urls.txt --output-dir results/ --llm-model gpt-4o
  
  # Use Anthropic Claude
  python -m signature_extractor_v3.main csv --source data.csv --preset anthropic
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
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create configuration
    if args.preset:
        config = ExtractionConfig.create_preset(
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
        config = ExtractionConfig(
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
    print(f"Model used: {config.llm.provider} - {config.llm.model}")
    
    return 0 if successful == total_pdfs else 1

if __name__ == "__main__":
    sys.exit(main())