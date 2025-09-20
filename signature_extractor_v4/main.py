import argparse
import sys
import os
from pathlib import Path
from .config import (
    ExtractionConfig, LLMConfig, ProcessingConfig, 
    get_all_presets, get_commercial_presets, get_opensource_presets,
    validate_config
)
from .core.orchestrator import ExtractionOrchestrator
from .adapters.csv_adapter import CSVSourceAdapter
from .adapters.directory_adapter import DirectorySourceAdapter
from .adapters.url_adapter import URLSourceAdapter
from .utils.logging_config import setup_logging

def print_preset_info():
    """Print information about available presets"""
    print("\nAvailable Model Presets:")
    print("=" * 50)
    
    # Get recommendations
    recommendations = ExtractionConfig.get_preset_recommendations()
    
    commercial = get_commercial_presets()
    opensource = get_opensource_presets()
    
    print("\nCommercial APIs:")
    print("-" * 20)
    for preset in commercial:
        print(f"  {preset}")
    
    print("\nOpen-Source Models:")
    print("-" * 20)
    for preset in opensource:
        print(f"  {preset}")
    
    print("\nRecommended Configurations:")
    print("-" * 30)
    for use_case, info in recommendations.items():
        print(f"\n{use_case.replace('_', ' ').title()}:")
        print(f"  Preset: {info['preset']}")
        print(f"  Description: {info['description']}")
        print(f"  Requirements: {info['requirements']}")
        print(f"  Performance: {info['expected_performance']}")

def check_dependencies(config: ExtractionConfig):
    """Check and warn about missing dependencies"""
    warnings = validate_config(config)
    
    if warnings:
        print("\nConfiguration Warnings:")
        for warning in warnings:
            print(f"  Warning: {warning}")
    
    # Check for specific dependencies based on provider
    missing_deps = []
    
    if config.llm.provider == "huggingface":
        try:
            import transformers
        except ImportError:
            missing_deps.append("transformers")
        
        try:
            import torch
        except ImportError:
            missing_deps.append("torch")
    
    if config.llm.provider == "vllm":
        try:
            import vllm
        except ImportError:
            missing_deps.append("vllm")
    
    if config.llm.provider == "ollama":
        import requests
        try:
            response = requests.get(f"{config.llm.ollama_base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"  Warning: Cannot connect to Ollama at {config.llm.ollama_base_url}")
        except:
            print(f"  Warning: Ollama not accessible at {config.llm.ollama_base_url}")
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="PDF Signature Extractor v3 - Enhanced with Open-Source VLM support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available model presets
  python -m signature_extractor_v3.main --list-presets
  
  # Auto-configure based on hardware
  python -m signature_extractor_v3.main csv --source statements.csv --preset auto
  
  # Use open-source model (CPU)
  python -m signature_extractor_v3.main csv --source statements.csv --preset cpu-smolvlm
  
  # Use open-source model (GPU)
  python -m signature_extractor_v3.main directory --source /path/to/pdfs/ --preset qwen-small
  
  # Use commercial API (existing behavior)
  python -m signature_extractor_v3.main csv --source data.csv --preset openai
  
  # Test model before running
  python -m signature_extractor_v3.main --test-model --preset minicpm
  
  # Custom open-source configuration
  python -m signature_extractor_v3.main csv --source data.csv \\
    --llm-provider huggingface \\
    --llm-model "Qwen/Qwen2.5-VL-7B-Instruct" \\
    --device cuda \\
    --quantization 4bit
        """
    )
    
    # Help and information options
    parser.add_argument("--list-presets", action="store_true",
                       help="List all available model presets and exit")
    parser.add_argument("--test-model", action="store_true",
                       help="Test model loading and basic inference")
    
    # Source configuration (optional for test mode)
    parser.add_argument("source_type", nargs='?', choices=["csv", "directory", "urls"],
                       help="Type of PDF source")
    parser.add_argument("--source", 
                       help="Path to source (CSV file, directory, or URL list)")
    
    # Model configuration
    parser.add_argument("--preset", default="auto",
                       help="Model preset to use (use --list-presets to see options)")
    
    # Manual LLM configuration (overrides preset)
    parser.add_argument("--llm-provider", 
                       choices=["openai", "anthropic", "huggingface", "vllm", "ollama", "local"],
                       help="LLM provider (overrides preset)")
    parser.add_argument("--llm-model",
                       help="LLM model name (overrides preset)")
    parser.add_argument("--api-key", help="API key for commercial providers")
    
    # Open-source specific options
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
                       help="Device for open-source models")
    parser.add_argument("--quantization", choices=["4bit", "8bit"],
                       help="Quantization for open-source models")
    parser.add_argument("--torch-dtype", choices=["float16", "float32", "bfloat16"],
                       help="Torch dtype for open-source models")
    
    # Output configuration
    parser.add_argument("--output-dir", default="signature_results",
                       help="Output directory for results")
    
    # Processing configuration
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of parallel workers")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for PDF to image conversion")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for processing")
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")
    
    args = parser.parse_args()
    
    # Handle special actions
    if args.list_presets:
        print_preset_info()
        return 0
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create configuration
    if args.preset and args.preset != "auto" and not args.llm_provider:
        # Use preset
        config = ExtractionConfig.create_preset(args.preset, args.output_dir)
    elif args.preset == "auto":
        # Auto-configure
        config = ExtractionConfig.auto_configure(output_dir=args.output_dir)
        print(f"Auto-configured to use preset based on detected hardware")
    else:
        # Manual configuration
        llm_config = LLMConfig(
            provider=args.llm_provider or "openai",
            model=args.llm_model or "gpt-4o",
            api_key=args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
            device=args.device or "auto",
            quantization=args.quantization,
            torch_dtype=args.torch_dtype or "float16"
        )
        
        config = ExtractionConfig(
            llm=llm_config,
            processing=ProcessingConfig(
                output_dir=args.output_dir,
                dpi=args.dpi,
                max_workers=args.max_workers,
                batch_size=args.batch_size
            )
        )
    
    # Check dependencies and configuration
    if not check_dependencies(config):
        return 1
    
    print(f"Using model: {config.llm.provider} - {config.llm.model}")
    
    # Test mode
    if args.test_model:
        print("\nTesting model loading and inference...")
        try:
            from .utils.llm_interface import LLMInterface
            from PIL import Image, ImageDraw
            
            # Create simple test
            llm_interface = LLMInterface(config.llm)
            print("Model loaded successfully")
            
            # Create test image
            test_img = Image.new('RGB', (400, 300), 'white')
            draw = ImageDraw.Draw(test_img)
            draw.text((50, 100), "Test Signature", fill='black')
            draw.text((50, 120), "John Doe", fill='black')
            draw.text((50, 140), "2024-03-15", fill='black')
            
            # Test inference
            test_prompt = "Do you see any text that looks like a signature? Keep response brief."
            import time
            start_time = time.time()
            response = llm_interface.process_image_with_prompt(test_img, test_prompt)
            end_time = time.time()
            
            print(f"Inference successful ({end_time - start_time:.2f}s)")
            print(f"Response: {response[:100]}...")
            print("\nModel is ready for document processing")
            return 0
            
        except Exception as e:
            print(f"Model test failed: {e}")
            return 1
    
    # Regular processing mode - require source
    if not args.source_type or not args.source:
        parser.error("Source type and source path are required for processing mode")
    
    # Create appropriate adapter
    if args.source_type == "csv":
        adapter = CSVSourceAdapter(config, args.source)
    elif args.source_type == "directory":
        adapter = DirectorySourceAdapter(config, args.source)
    else:  # urls
        adapter = URLSourceAdapter(config, args.source)
    
    try:
        # Process sources
        orchestrator = ExtractionOrchestrator(config)
        sources = adapter.get_pdf_sources()
        
        print(f"Found {len(sources)} sources to process")
        
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
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())