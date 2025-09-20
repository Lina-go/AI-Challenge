# setup_open_source_vlms.py
"""
Setup script for signature extractor with open-source VLM support.
Helps users install the right dependencies for their hardware and use case.
"""

import sys
import subprocess
import platform
import argparse
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def check_system_requirements():
    """Check system requirements"""
    print("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print("Error: Python 3.8+ required")
        return False
    else:
        print(f"Python {python_version.major}.{python_version.minor}: OK")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"System RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 8:
            print("Warning: Less than 8GB RAM may limit model options")
    except ImportError:
        print("Warning: Cannot check memory (psutil not installed)")
    
    # Check CUDA
    cuda_available = check_cuda()
    if cuda_available:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        except:
            print("CUDA available")
    else:
        print("No CUDA GPU detected - will use CPU-optimized models")
    
    return True

def install_base_requirements():
    """Install base requirements"""
    print("\nInstalling base requirements...")
    
    base_packages = [
        "pandas>=1.5.0",
        "pillow>=9.0.0", 
        "requests>=2.28.0",
        "PyMuPDF>=1.20.0",
        "psutil>=5.9.0"
    ]
    
    for package in base_packages:
        run_command(f"pip install {package}")

def install_torch(use_cuda=True):
    """Install PyTorch with appropriate CUDA support"""
    print(f"\nInstalling PyTorch (CUDA: {use_cuda})...")
    
    if use_cuda:
        # Install CUDA version
        run_command("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    else:
        # Install CPU version
        run_command("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

def install_transformers_stack():
    """Install transformers and related packages"""
    print("\nInstalling transformers stack...")
    
    packages = [
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "safetensors>=0.4.0",
        "optimum>=1.14.0"
    ]
    
    for package in packages:
        run_command(f"pip install {package}")

def install_quantization_support():
    """Install quantization libraries"""
    print("\nInstalling quantization support...")
    
    # Check if CUDA is available for bitsandbytes
    cuda_available = check_cuda()
    
    if cuda_available:
        run_command("pip install bitsandbytes>=0.41.0")
        print("Installed bitsandbytes with CUDA support")
    else:
        print("Skipping bitsandbytes (requires CUDA)")

def install_flash_attention():
    """Install flash attention (CUDA only)"""
    print("\nInstalling flash attention...")
    
    cuda_available = check_cuda()
    
    if cuda_available:
        try:
            # Flash attention requires specific setup
            run_command("pip install flash-attn>=2.3.0 --no-build-isolation")
            print("Installed flash attention")
        except:
            print("Flash attention installation failed (optional)")
    else:
        print("Skipping flash attention (requires CUDA)")

def install_vllm():
    """Install vLLM for high-performance inference"""
    print("\nInstalling vLLM...")
    
    cuda_available = check_cuda()
    
    if cuda_available:
        try:
            run_command("pip install vllm>=0.2.7")
            print("Installed vLLM")
        except:
            print("vLLM installation failed (optional)")
    else:
        print("Skipping vLLM (requires CUDA)")

def install_commercial_apis():
    """Install commercial API clients"""
    print("\nInstalling commercial API clients...")
    
    packages = [
        "openai>=1.0.0",
        "anthropic>=0.7.0"
    ]
    
    for package in packages:
        run_command(f"pip install {package}")

def install_ollama_support():
    """Install Ollama Python client"""
    print("\nInstalling Ollama support...")
    run_command("pip install ollama-python>=0.1.0")

def setup_environment_file():
    """Create a template environment file"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("\nCreating environment file template...")
        
        env_content = """# API Keys for commercial providers (optional)
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here

# HuggingFace settings (optional)
# HF_TOKEN=your_huggingface_token_here
# HF_HOME=/path/to/huggingface/cache

# Ollama settings (if using Ollama)
# OLLAMA_BASE_URL=http://localhost:11434

# Hardware optimization
# CUDA_VISIBLE_DEVICES=0
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"Created {env_file}")
        print("Edit this file to configure API keys and settings")

def main():
    parser = argparse.ArgumentParser(description="Setup signature extractor with open-source VLM support")
    parser.add_argument("--profile", choices=["minimal", "standard", "full", "commercial-only"],
                       default="standard", help="Installation profile")
    parser.add_argument("--cpu-only", action="store_true", help="Install CPU-only versions")
    parser.add_argument("--skip-cuda-check", action="store_true", help="Skip CUDA availability check")
    parser.add_argument("--no-flash-attention", action="store_true", help="Skip flash attention installation")
    parser.add_argument("--no-vllm", action="store_true", help="Skip vLLM installation")
    
    args = parser.parse_args()
    
    print("Setting up Signature Extractor with Open-Source VLM support")
    print("=" * 60)
    
    # Check requirements
    if not check_system_requirements():
        sys.exit(1)
    
    # Determine CUDA usage
    use_cuda = not args.cpu_only and (args.skip_cuda_check or check_cuda())
    
    print(f"\nInstallation profile: {args.profile}")
    print(f"CUDA support: {use_cuda}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    
    # Install based on profile
    if args.profile == "commercial-only":
        install_base_requirements()
        install_commercial_apis()
    
    elif args.profile == "minimal":
        install_base_requirements()
        install_torch(use_cuda)
        install_transformers_stack()
        if use_cuda:
            install_quantization_support()
    
    elif args.profile == "standard":
        install_base_requirements()
        install_torch(use_cuda)
        install_transformers_stack()
        install_commercial_apis()
        
        if use_cuda:
            install_quantization_support()
            if not args.no_flash_attention:
                install_flash_attention()
    
    elif args.profile == "full":
        install_base_requirements()
        install_torch(use_cuda)
        install_transformers_stack()
        install_commercial_apis()
        install_ollama_support()
        
        if use_cuda:
            install_quantization_support()
            if not args.no_flash_attention:
                install_flash_attention()
            if not args.no_vllm:
                install_vllm()
    
    # Setup environment
    setup_environment_file()
    
    print("\nInstallation complete")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys (if using commercial models)")
    print("2. Test the installation:")
    print("   python test_open_source_vlms.py --quick")
    print("3. Run a quick signature extraction:")
    print("   python -m signature_extractor_v3.main --test-model --preset auto")
    
    print("\nRecommended presets for your system:")
    if use_cuda:
        print("  - Fast & accurate: qwen-small")
        print("  - Balanced: minicpm") 
        print("  - Lightweight: smolvlm")
    else:
        print("  - CPU optimized: cpu-smolvlm")
        print("  - Ultra lightweight: edge-tiny")
    
    print("\nFor more options: python -m signature_extractor_v3.main --list-presets")

if __name__ == "__main__":
    main()