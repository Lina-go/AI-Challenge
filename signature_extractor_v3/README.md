# Signature Extractor

LLM-signature extraction from PDF documents with modular architecture and comprehensive output.

## Architecture

```
signature_extractor_v3/
├── config.py                # Configuration classes
├── base_extractor.py        # Abstract base classes
├── main.py                  # CLI interface
├── adapters/                # PDF source adapters
│   ├── csv_adapter.py       # CSV file processing
│   ├── directory_adapter.py # Directory scanning
│   └── url_adapter.py       # URL list processing
├── core/                    # Core processing components
│   ├── orchestrator.py      # Main workflow coordinator
│   ├── document_processor.py# PDF to image conversion
│   ├── signature_analyzer.py# Optimized LLM-based analysis (2 prompts)
│   ├── data_processor.py    # Data cleaning & formatting
│   └── result_manager.py    # Output management
├── models/                  # Data structures (pending)
├── prompts/                 # Optimized LLM prompt templates
│   ├── signature_detector_analyzer.txt  # Detection + characteristics
│   └── signature_data_extractor.txt     # Text + date extraction
├── utils/                   # Utility functions
│   ├── progress_tracker.py  # Progress monitoring
│   ├── llm_interface.py     # LLM provider abstraction
│   ├── prompt_loader.py     # Prompt management
│   └── logging_config.py    # Logging setup
└── configs/                 # Configuration templates (pending)
```

**Output Columns (Matches signature_columns.xlsx):**
1. `Signature` - Yes/No signature presence
2. `Signature_Yes_text` - Signatory name, title, company
3. `Signature_Image` - Yes/No if signature is an image
4. `Signature_scanned` - Yes/No if signature appears scanned
5. `Presence_Signature_date` - Yes/No date presence
6. `Signature_Date` - Actual signature date

## Quick Start

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 2. Setup API Key
```bash
# OpenAI (recommended)
export OPENAI_API_KEY="your-openai-key"

# Or Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 3. Test Installation
```bash
python test_signature_extractor.py
```

## Usage

### CLI Interface

```bash
# Extract from CSV file
python -m signature_extractor_v2.main csv --source statements.csv

# Extract from directory
python -m signature_extractor_v2.main directory --source /path/to/pdfs/

# Extract from URL list
python -m signature_extractor_v2.main urls --source urls.txt

# Custom configuration
python -m signature_extractor_v2.main csv \
  --source data.csv \
  --output-dir results/ \
  --llm-provider anthropic \
  --llm-model claude-3-sonnet \
  --max-workers 2
```

### Programmatic Usage

```python
from signature_extractor_v2 import extract_signatures_from_csv

# Simple usage
results = extract_signatures_from_csv("statements.csv", "output/")

# Advanced usage
from signature_extractor_v2 import (
    ExtractionConfig, LLMConfig, ProcessingConfig,
    ExtractionOrchestrator, CSVSourceAdapter
)

config = ExtractionConfig(
    llm=LLMConfig(provider="openai", model="gpt-4o"),
    processing=ProcessingConfig(output_dir="results/")
)

adapter = CSVSourceAdapter(config, "data.csv")
orchestrator = ExtractionOrchestrator(config)

sources = adapter.get_pdf_sources()
for source in sources:
    source['pdf_path'] = adapter.prepare_source(source)

results = orchestrator.process_multiple_sources(sources)
```

## Input Formats

### CSV Format
```csv
pdf_url,document_name
https://example.com/doc1.pdf,Document 1
/local/path/doc2.pdf,Document 2
```

### URL List Format
```
# URLs to PDF documents
https://example.com/statement1.pdf
https://example.com/statement2.pdf
# Comments start with #
https://example.com/statement3.pdf
```

## Output

The system generates:
- **Consolidated CSV**: All signatures with required columns
- **Individual Results**: Per-source JSON and CSV files
- **Processing Summary**: Success/failure statistics
- **Summary Report**: Human-readable extraction overview

### Output Directory Structure
```
signature_results/
├── csv_results/                    # Consolidated CSV files
│   └── signature_extraction_results_YYYYMMDD_HHMMSS.csv
├── json_results/                   # Consolidated JSON files
│   └── signature_extraction_results_YYYYMMDD_HHMMSS.json
├── individual_results/             # Per-source results
│   ├── source_id_results.json
│   └── source_id_results.csv
├── page_images/                    # Extracted page images (if enabled)
├── downloads/                      # Downloaded PDFs (for URL sources)
├── processing_summary_*.csv        # Processing statistics
└── extraction_report_*.txt         # Summary reports
```

## Configuration

### LLM Providers
- **OpenAI**: GPT-4o (recommended), GPT-4-turbo (with vision)
- **Anthropic**: Claude-3-sonnet, Claude-3-opus (with vision)

## Requirements

```
pandas>=1.5.0
pillow>=9.0.0
requests>=2.28.0
PyMuPDF>=1.20.0
openai>=1.0.0          # For OpenAI models
anthropic>=0.7.0       # For Anthropic models
```

## Testing

```bash
# Run basic functionality tests
python test_signature_extractor.py

# Test with API (requires API key)
export OPENAI_API_KEY="your-key"
python test_signature_extractor.py

# Test with different sources
python -m signature_extractor_v2.main directory --source test_pdfs/
```

### Debug Mode
```bash
python -m signature_extractor_v2.main csv --source data.csv --log-level DEBUG
```