# PDF Processor

Document analysis and signature extraction framework with LLM content extraction from PDF documents

## Architecture

```
pdf_processor/
├── __init__.py                # Main module with convenience functions
├── config.py                  # Configuration classes (ProcessorConfig, LLMConfig, etc.)
├── main.py                    # CLI interface
├── requirements.txt           # Dependencies
├── adapters/                  # PDF source adapters
│   ├── base_adapter.py        # Abstract base class
│   ├── csv_adapter.py         # CSV file processing
│   ├── directory_adapter.py   # Directory scanning
│   └── url_adapter.py         # URL list processing
├── core/                      # Core processing components
│   ├── document_processor.py  # Main document processing workflow
│   ├── llm_interface.py       # Unified LLM interface (OpenAI, Anthropic, etc.)
│   ├── page_processor.py      # Page-level analysis and extraction
│   ├── orchestrator.py        # Main workflow coordinator
│   ├── signature_analyzer.py  # LLM-based signature detection
│   ├── data_processor.py      # Data cleaning & formatting
│   ├── result_manager.py      # Output management
│   └── llm_factory.py         # LLM provider factory
├── models/                    # Data structures
│   ├── document_result.py     # Document processing results
│   ├── page_analysis.py       # Page content analysis results
│   └── page_content.py        # Extracted page content container
├── prompts/                   # Optimized LLM prompt templates
│   ├── page_content_analyzer.txt    # Page content type analysis
│   ├── text_extractor.txt           # Text extraction with placeholders
│   ├── table_extractor.txt          # Structured table extraction
│   ├── figure_extractor.txt         # Visual content extraction
│   ├── signature_detector_analyzer.txt  # Signature detection + analysis
│   └── signature_data_extractor.txt     # Signature text + date extraction
├── utils/                     # Utility functions
│   ├── pdf_converter.py       # PDF to image conversion (PyMuPDF)
│   ├── prompt_loader.py       # Prompt template management
│   ├── content_parser.py      # LLM response parsing
│   ├── content_integrator.py  # Content integration with placeholders
│   ├── progress_tracker.py    # Progress monitoring
│   └── logging_config.py      # Logging configuration
└── outputs/                   # Example outputs
    └── test_output.md         # Sample processed document
```

**Dual Processing Modes:**
1. **Document Processing**: Extract text, tables, figures → Clean markdown output
2. **Signature Extraction**: Detect signatures → Structured CSV with signature data

**Output Formats:**
- **Document Processing**: Integrated markdown with tables, figures, and placeholders
- **Signature Extraction**: CSV with columns matching signature analysis requirements

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone <repository-url>
cd pdf_processor

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 2. Setup API Keys
```bash
# OpenAI (default - recommended)
export OPENAI_API_KEY="your-openai-key"

# Anthropic (alternative)
export ANTHROPIC_API_KEY="your-anthropic-key"

# HuggingFace (for open models)
export HF_TOKEN="your-huggingface-token"
```

### 3. Test Installation
```bash
# Test with a single document
python -m pdf_processor_v2.main document single test.pdf --output test_output.md
```

## Usage

### CLI Interface

#### Document Processing (Text, Tables, Figures)
```bash
# Process single PDF
python -m pdf_processor_v2.main document single input.pdf --output output.md

# Process directory of PDFs
python -m pdf_processor_v2.main document directory /path/to/pdfs/ --output-dir results/

# Use different LLM providers
python -m pdf_processor_v2.main document single input.pdf --preset anthropic
python -m pdf_processor_v2.main document single input.pdf --preset glm-4.5v
```

#### Signature Extraction
```bash
# Extract from CSV file
python -m pdf_processor_v2.main signature csv statements.csv --output-dir signature_results/

# Extract from directory
python -m pdf_processor_v2.main signature directory /path/to/pdfs/ --preset ollama-moondream

# Extract from URL list
python -m pdf_processor_v2.main signature urls urls.txt --output-dir results/
```

#### Advanced Configuration
```bash
# Custom LLM settings
python -m pdf_processor_v2.main document single input.pdf \
  --llm-provider huggingface \
  --llm-model "Qwen/Qwen2.5-VL-7B-Instruct" \
  --base-url "https://router.huggingface.co/v1"

# Local Ollama models
python -m pdf_processor_v2.main signature directory pdfs/ \
  --llm-provider ollama \
  --llm-model "moondream:1.8b" \
  --ollama-url "http://localhost:11434"
```

### Programmatic Usage

#### Simple Document Processing
```python
from pdf_processor_v2 import process_document

# Basic usage
result = process_document("document.pdf", "output.md")

# With custom LLM
result = process_document(
    "document.pdf", 
    "output.md",
    llm_provider="anthropic",
    llm_model="claude-3-sonnet"
)
```

#### Advanced Usage
```python
from pdf_processor_v2 import (
    ProcessorConfig, LLMConfig, ProcessingConfig, TaskConfig,
    DocumentProcessor, create_llm_interface
)

# Create configuration
config = ProcessorConfig(
    llm=LLMConfig(provider="openai", model="gpt-4o"),
    processing=ProcessingConfig(output_dir="results/", dpi=300),
    task=TaskConfig(extract_tables=True, extract_figures=True)
)

# Process document
llm = create_llm_interface(config.llm)
processor = DocumentProcessor(config)
result = processor.process_document("document.pdf")

print(f"Processed {result.processed_pages}/{result.total_pages} pages")
print(f"Success rate: {result.get_success_rate():.1f}%")
```

#### Signature Extraction
```python
from pdf_processor_v2 import extract_signatures_from_csv

# Simple signature extraction
results = extract_signatures_from_csv("statements.csv", "signature_results/")

# Advanced configuration
from pdf_processor_v2 import (
    ProcessorConfig, CSVSourceAdapter, SignatureOrchestrator
)

config = ProcessorConfig.create_signature_preset("openai", "results/")
adapter = CSVSourceAdapter(config, "data.csv")
orchestrator = SignatureOrchestrator(config)

sources = adapter.get_pdf_sources()
for source in sources:
    source['pdf_path'] = adapter.prepare_source(source)

results = orchestrator.process_multiple_sources(sources)
```

## Input Formats

### CSV Format (for batch processing)
```csv
pdf_url,document_name
https://example.com/doc1.pdf,Annual Report 2024
/local/path/statement.pdf,Financial Statement
```

### URL List Format
```
# PDF document URLs
https://example.com/report1.pdf
https://example.com/statement1.pdf
# Comments start with #
https://example.com/disclosure1.pdf
```

## LLM Provider Support

### Commercial APIs
- **OpenAI**: `gpt-4o` (recommended), `gpt-4-turbo` with vision
- **Anthropic**: `claude-3-sonnet`, `claude-3-opus` with vision

### Open Source Models
- **HuggingFace Router**: `GLM-4.5V`, `Qwen2.5-VL-7B-Instruct`
- **Ollama (Local)**: `moondream:1.8b`, `llava:7b`, `llava:13b`

### Preset Configurations
```bash
# Commercial presets
--preset openai           # GPT-4o
--preset anthropic        # Claude-3-sonnet

# Open source presets  
--preset glm-4.5v         # HuggingFace GLM-4.5V
--preset qwen-vl          # HuggingFace Qwen2.5-VL
--preset ollama-moondream # Local Ollama Moondream
--preset ollama-llava     # Local Ollama LLaVA
```

## Output

### Document Processing Output
- **Markdown files**: Clean, structured content with integrated tables and figures
- **Placeholder system**: `[TABLE_1]`, `[FIGURE_2]`, `[SIGNATURE_1]` for content integration
- **Processing statistics**: Page success rates, content type counts

### Signature Extraction Output
- **Consolidated CSV**: All signatures with standardized columns
- **Individual Results**: Per-document JSON and CSV files  
- **Processing Summary**: Success/failure statistics
- **Summary Report**: Human-readable extraction overview

#### Output Directory Structure
```
results/
├── document_content.md              # Processed document (document mode)
├── csv_results/                     # Consolidated results (signature mode)
│   └── signature_extraction_results_YYYYMMDD_HHMMSS.csv
├── json_results/                    # Consolidated JSON
│   └── signature_extraction_results_YYYYMMDD_HHMMSS.json
├── individual_results/              # Per-source results
│   ├── source_id_results.json
│   └── source_id_results.csv
├── downloads/                       # Downloaded PDFs (URL sources)
├── processing_summary_*.csv         # Processing statistics
└── extraction_report_*.txt          # Summary reports
```

### Signature Extraction Columns
1. `Signature` - Yes/No signature presence
2. `Signature_Yes_text` - Signatory name, title, company information
3. `Signature_Image` - Yes/No if signature appears as image
4. `Signature_scanned` - Yes/No if signature appears scanned/photographed
5. `Presence_Signature_date` - Yes/No date presence near signature
6. `Signature_Date` - Extracted and standardized signature date

## Content Extraction Features

### Text Processing
- **Smart text extraction** with structure preservation
- **Placeholder integration** for tables, figures, signatures
- **Markdown formatting** with headers, lists, emphasis

### Table Extraction  
- **Structured data detection** in tabular format
- **Markdown table conversion** with proper formatting
- **Multi-row content support** with line breaks

### Figure Extraction
- **Visual content analysis**: Charts, diagrams, organizational structures
- **Semantic description**: Extract meaning from visual elements
- **Hierarchy preservation**: Organizational charts, process flows

### Signature Detection
- **Multi-modal analysis**: Handwritten, typed, digital signatures
- **Contextual extraction**: Names, titles, dates associated with signatures
- **Confidence scoring**: High/medium/low detection confidence

## Requirements

```
# Core dependencies
PyMuPDF>=1.23.0        # PDF processing
Pillow>=10.0.0          # Image handling
pandas>=2.0.0           # Data processing
requests>=2.31.0        # HTTP requests

# LLM providers
openai>=1.0.0           # OpenAI GPT models
anthropic>=0.18.0       # Anthropic Claude models

# Optional: Open source models
transformers>=4.30.0    # HuggingFace transformers
torch>=2.0.0           # PyTorch for local models
ollama>=0.1.0          # Ollama local models

# Utilities
pathlib                # Path handling
typing-extensions      # Type hints
```

## Testing & Debugging

### Basic Testing
```bash
# Test single document
python -m pdf_processor_v2.main document single test.pdf --output test_output.md

# Test with debug logging
python -m pdf_processor_v2.main document single test.pdf --log-level DEBUG

# Test different providers
python -m pdf_processor_v2.main document single test.pdf --preset anthropic
```

### Debug Mode Features
- **Page-by-page logging**: Detailed processing steps
- **Prompt response saving**: Debug LLM interactions  
- **Content integration validation**: Placeholder resolution tracking
- **Error handling**: Graceful failure with error reporting

### Performance Testing
```bash
# Process multiple documents
python -m pdf_processor_v2.main document directory test_pdfs/ --max-workers 4

# Monitor processing speed
python -m pdf_processor_v2.main signature csv large_dataset.csv --log-level INFO
```

## Configuration Examples

### Local Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull vision model
ollama pull moondream:1.8b

# Process with local model
python -m pdf_processor_v2.main document single doc.pdf --preset ollama-moondream
```

### HuggingFace Setup
```bash
# Set token
export HF_TOKEN="your-huggingface-token"

# Use HuggingFace Router
python -m pdf_processor_v2.main document single doc.pdf --preset glm-4.5v
```

## Use Cases

- **Financial Reports**: Extract structured data and signatures from annual reports
- **Legal Documents**: Process contracts with signature verification
- **Modern Slavery Statements**: Comprehensive analysis with signature extraction
- **Academic Papers**: Extract text, tables, figures with proper formatting
- **Compliance Documents**: Automated signature detection for audit trails

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

[Add your license information here]