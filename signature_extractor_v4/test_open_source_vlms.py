"""
Test script for open-source VLM integration with signature extractor.
This script helps validate that models work correctly before full deployment.
"""

import time
import logging
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import torch

from signature_extractor_v4.config import ExtractionConfig, validate_config
from signature_extractor_v4.utils.llm_interface import LLMInterface

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_image():
    """Create a simple test image with signature-like elements"""
    # Create a 800x600 white image
    img = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw some text that looks like a signature block
    draw.text((50, 400), "John Smith", fill='black')
    draw.text((50, 420), "Chief Executive Officer", fill='black')
    draw.text((50, 440), "ABC Corporation", fill='black')
    draw.text((50, 460), "Date: March 15, 2024", fill='black')
    
    # Draw a signature-like scribble
    points = [(200, 380), (220, 370), (240, 385), (260, 375), (280, 390), (300, 380)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill='blue', width=2)
    
    return img

def test_model_loading(preset: str):
    """Test if a model can be loaded successfully"""
    logger.info(f"Testing model loading for preset: {preset}")
    
    try:
        config = ExtractionConfig.create_preset(preset)
        warnings = validate_config(config)
        
        if warnings:
            logger.warning(f"Configuration warnings: {warnings}")
        
        # Try to create the interface
        llm_interface = LLMInterface(config.llm)
        
        # Get model info
        model_info = llm_interface.get_model_info()
        logger.info(f"Model loaded successfully: {model_info}")
        
        return True, llm_interface
        
    except Exception as e:
        logger.error(f"Failed to load model {preset}: {e}")
        return False, None

def test_inference(llm_interface: LLMInterface, test_image: Image.Image):
    """Test inference with a simple prompt"""
    logger.info("Testing inference...")
    
    simple_prompt = """
    Look at this image and tell me:
    1. Do you see any signatures or signature-like elements?
    2. Do you see any names or dates?
    
    Keep your response brief.
    """
    
    try:
        start_time = time.time()
        response = llm_interface.process_image_with_prompt(test_image, simple_prompt)
        inference_time = time.time() - start_time
        
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        logger.info(f"Response: {response[:200]}...")
        
        return True, inference_time, response
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return False, None, None

def benchmark_preset(preset: str, num_runs: int = 3):
    """Benchmark a specific preset"""
    logger.info(f"Benchmarking preset: {preset}")
    
    # Test model loading
    success, llm_interface = test_model_loading(preset)
    if not success:
        return None
    
    # Create test image
    test_image = create_test_image()
    
    # Run multiple inference tests
    inference_times = []
    for i in range(num_runs):
        logger.info(f"Run {i + 1}/{num_runs}")
        success, inference_time, response = test_inference(llm_interface, test_image)
        if success:
            inference_times.append(inference_time)
        else:
            logger.warning(f"Run {i + 1} failed")
    
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        
        results = {
            'preset': preset,
            'successful_runs': len(inference_times),
            'total_runs': num_runs,
            'avg_inference_time': avg_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'model_info': llm_interface.get_model_info()
        }
        
        logger.info(f"Benchmark results for {preset}:")
        logger.info(f"  Success rate: {len(inference_times)}/{num_runs}")
        logger.info(f"  Average time: {avg_time:.2f}s")
        logger.info(f"  Min/Max time: {min_time:.2f}s / {max_time:.2f}s")
        
        return results
    
    return None

def get_system_info():
    """Get system information for benchmarking context"""
    import psutil
    
    info = {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'cuda_available': torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info

def main():
    parser = argparse.ArgumentParser(description="Test open-source VLM integration")
    parser.add_argument("--preset", type=str, help="Specific preset to test")
    parser.add_argument("--benchmark", action="store_true", help="Run comprehensive benchmark")
    parser.add_argument("--cpu-only", action="store_true", help="Test only CPU-compatible presets")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer runs")
    parser.add_argument("--save-results", type=str, help="Save results to file")
    
    args = parser.parse_args()
    
    # Print system info
    logger.info("System Information:")
    system_info = get_system_info()
    for key, value in system_info.items():
        logger.info(f"  {key}: {value}")
    
    # Determine which presets to test
    if args.preset:
        presets_to_test = [args.preset]
    elif args.cpu_only:
        presets_to_test = ["cpu-florence", "cpu-minicpm", "cpu-qwen", "edge-florence"]
    elif args.benchmark:
        # Test a representative sample
        if system_info['cuda_available']:
            presets_to_test = [
                "qwen-small", "minicpm", "smolvlm", "granite-docling",
                "cpu-smolvlm", "florence-base"
            ]
        else:
            presets_to_test = ["cpu-smolvlm", "cpu-llava-tiny", "cpu-granite"]
    else:
        # Quick test with auto-configuration
        config = ExtractionConfig.auto_configure()
        presets_to_test = [config.llm.model]  # This won't work as expected, need to fix
        # For now, use a safe default
        if system_info['cuda_available']:
            presets_to_test = ["minicpm"]
        else:
            presets_to_test = ["cpu-florence"]

    # Run tests
    num_runs = 1 if args.quick else 3
    all_results = []
    
    for preset in presets_to_test:
        try:
            logger.info(f"\nTesting preset: {preset}")
            logger.info("=" * 50)
            
            results = benchmark_preset(preset, num_runs)
            if results:
                all_results.append(results)
            else:
                logger.error(f"Preset {preset} failed completely")
                
        except KeyboardInterrupt:
            logger.info("Testing interrupted by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error testing {preset}: {e}")
    
    # Print summary
    logger.info("\nSUMMARY")
    logger.info("=" * 50)
    
    if all_results:
        # Sort by average inference time
        all_results.sort(key=lambda x: x['avg_inference_time'])
        
        logger.info(f"{'Preset':<20} {'Success':<10} {'Avg Time':<12} {'Device':<10}")
        logger.info("-" * 60)
        
        for result in all_results:
            success_rate = f"{result['successful_runs']}/{result['total_runs']}"
            avg_time = f"{result['avg_inference_time']:.2f}s"
            device = result['model_info'].get('actual_device', 'unknown')
            
            logger.info(f"{result['preset']:<20} {success_rate:<10} {avg_time:<12} {device:<10}")
        
        # Recommendations
        logger.info("\nRecommendations:")
        fastest = all_results[0]
        logger.info(f"Fastest: {fastest['preset']} ({fastest['avg_inference_time']:.2f}s avg)")
        
        most_reliable = max(all_results, key=lambda x: x['successful_runs'])
        logger.info(f"Most reliable: {most_reliable['preset']} ({most_reliable['successful_runs']}/{most_reliable['total_runs']} success)")
        
    else:
        logger.error("No successful tests completed")
    
    # Save results if requested
    if args.save_results and all_results:
        import json
        
        save_data = {
            'system_info': system_info,
            'test_results': all_results,
            'timestamp': time.time()
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Results saved to {args.save_results}")

if __name__ == "__main__":
    main()