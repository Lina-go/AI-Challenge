# PDF Signature Extractor

Signature extractor from PDF documents using YOLOv8 object detection.

## Architecture Overview

The framework follows the following architecture:

```
signature_extractor/
├── __init__.py              # Package initialization and exports
├── config.py                # Configuration management
├── base_extractor.py        # Abstract base classes
├── main.py                  # CLI entry point
├── core/                    # Core processing components
│   ├── orchestrator.py      # Workflow coordination
│   ├── pdf_processor.py     # PDF to image conversion
│   ├── model_manager.py     # YOLO model management
│   ├── signature_detector.py # Signature detection logic
│   ├── image_processor.py   # Image cropping and manipulation
│   └── result_manager.py    # Results collection and saving
├── utils/                   # Utility modules
│   ├── download.py          # PDF download utilities
│   ├── progress.py          # Progress tracking
│   └── logging.py           # Logging configuration
├── adapters/                # Source-specific implementations
│   ├── csv_adapter.py       # Process PDFs from CSV files
│   └── directory_adapter.py # Process PDFs from directories
├── configs/                 # Model configurations
│   └── yolo_config.py       # YOLO model registry
└── README.md               # This file
```

## Key Components

### 1. Configuration System
- **`config.py`**: Central configuration management
- **`configs/`**: Model-specific configurations
- **`ExtractorConfig`**: Dataclass containing detection and processing settings

### 2. Core Processing Engine (`core/`)
- **`orchestrator.py`**: Coordinates the complete extraction workflow
- **`pdf_processor.py`**: Converts PDFs to images
- **`model_manager.py`**: Downloads and manages YOLO models
- **`signature_detector.py`**: Runs YOLO detection and applies selection logic
- **`image_processor.py`**: Crops and saves signature images
- **`result_manager.py`**: Collects and saves extraction results

### 3. Source Adapters (`adapters/`)
- **CSV Adapter**: Process PDFs listed in CSV files with URLs
- **Directory Adapter**: Process all PDFs in a local directory

## Basic Usage

### CLI Interface
```bash
# Process PDFs from CSV file
python -m signature_extractor.main --source-type csv --source statements.csv

# Process all PDFs in directory
python -m signature_extractor.main --source-type directory --source /path/to/pdfs/

# Custom configuration
python -m signature_extractor.main --source-type csv --source data.csv \
    --output-dir results/ --conf-threshold 0.7
```

### Program use
```python
from signature_extractor import ExtractorConfig, DetectionConfig, ProcessingConfig
from signature_extractor.adapters import CSVSourceAdapter
from signature_extractor.core import ExtractionOrchestrator

# Create configuration
config = ExtractorConfig(
    detection=DetectionConfig(conf_threshold=0.6),
    processing=ProcessingConfig(output_dir="results/")
)

# Process PDFs
adapter = CSVSourceAdapter(config, "statements.csv")
orchestrator = ExtractionOrchestrator(config)

sources = adapter.get_pdf_sources()
for source in sources:
    source['pdf_path'] = adapter.prepare_source(source)

results = orchestrator.process_multiple_pdfs(sources)
```

## Configuration

### Detection Configuration
```python
DetectionConfig(
    model_name="tech4humans/yolov8s-signature-detector",
    conf_threshold=0.8,           # Confidence threshold
    ensure_one_per_page=False,     # Force at least one detection per page
    device="cpu"                  # Device for inference
)
```

## Extending the Framework

### Adding New Source Types

1. **Create adapter** in `adapters/new_source.py`:
```python
class NewSourceAdapter(BaseSourceAdapter):
    def get_pdf_sources(self) -> List[Dict[str, Any]]:
        # Return list of PDF sources
        pass
    
    def prepare_source(self, source: Dict[str, Any]) -> str:
        # Prepare/download PDF and return local path
        pass
```

2. **Add to main.py**:
```python
elif args.source_type == "new_source":
    adapter = NewSourceAdapter(config, args.source)
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Output

The framework generates:
- **Signature crops**: Individual signature images
- **JSON results**: Detailed detection results per PDF
- **Summary report**: Overall extraction statistics
- **Progress logs**: Real-time processing updates