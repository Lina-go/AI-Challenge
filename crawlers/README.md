# Modular Crawler Framework

A flexible, extensible framework for crawling modern slavery statements from different government repositories with support for multiple jurisdictions and country-based filtering.

## Architecture Overview

The framework follows a modular, single-responsibility principle design:

```
crawlers/
├── __init__.py              # Package initialization and exports
├── config.py                # Configuration management and jurisdiction registry
├── base_crawler.py          # Abstract base class defining crawler interface
├── main.py                  # CLI entry point with multi-jurisdiction support
├── core/                    # Core crawling components
│   ├── orchestrator.py      # Workflow coordination and high-level logic
│   ├── page_crawler.py      # HTTP requests and HTML content extraction
│   ├── url_generator.py     # URL generation and template management
│   └── data_processor.py    # Data transformation and validation
├── utils/                   # Focused utility modules
│   ├── http.py              # HTTP request handling with retry logic
│   ├── html.py              # HTML parsing utilities
│   ├── url.py               # URL manipulation utilities
│   ├── data.py              # Data processing and file I/O utilities
│   ├── logging.py           # Logging configuration and progress tracking
│   └── streaming_logger.py  # Incremental result logging
├── adapters/                # Site-specific implementations
│   ├── canadian.py          # Canadian Public Safety repository
│   └── australian.py        # Australian Modern Slavery Register
├── configs/                 # Jurisdiction-specific configurations
│   ├── canadian.py          # Canadian configuration factory
│   └── australian.py        # Australian configuration factory
└── README.md               # This file
```

## Key Components

### 1. Configuration System
- **`config.py`**: Central configuration management with hardcoded jurisdiction registry
- **`configs/`**: Jurisdiction-specific configuration factories that define crawling parameters
- **`CrawlerConfig`**: Dataclass containing all site-specific settings (URLs, pagination, delays, etc.)

### 2. Core Crawling Engine (`core/`)
The framework uses a modular core that separates different crawling concerns:
- **`orchestrator.py`**: Coordinates the complete crawl workflow and manages component interactions
- **`page_crawler.py`**: Handles HTTP requests, retry logic, and HTML content extraction
- **`url_generator.py`**: Generates search URLs based on configuration templates
- **`data_processor.py`**: Processes, cleans, and validates extracted data

### 3. Base Crawler Interface (`base_crawler.py`)
- **`BaseCrawler`**: Abstract base class that defines the crawler interface
- Provides a simple facade over the modular core components
- Site-specific adapters implement two key methods: `extract_statement_links()` and `extract_statement_data()`

### 4. Site-Specific Adapters (`adapters/`)
Each jurisdiction has its own adapter handling unique data structures:
- **Canadian**: HTML scraping with country detection for post-processing filtering
- **Australian**: Direct CSV download with built-in country filtering

### 5. Utility Modules (`utils/`)
Focused, reusable utilities following single-responsibility principle:
- HTTP handling, HTML parsing, URL manipulation, data I/O, logging, and streaming results

## Basic Usage

### CLI Interface (Recommended)
```bash
# List available jurisdictions
python -m crawlers.main --help

# Crawl all Canadian statements (saves to output/canadian/)
python -m crawlers.main canadian

# Crawl Australian statements for all countries (saves to output/australian/)
python -m crawlers.main australian --country all

# Filter by country (Canadian: pattern-based, Australian: exact)
python -m crawlers.main canadian --country "Netherlands"
python -m crawlers.main australian --country "United States of America"

# List available countries for filtering
python -m crawlers.main --list-countries canadian
python -m crawlers.main --list-countries australian
```

### Programmatic Usage
```python
from crawlers.adapters import CanadianAdapter, AustralianAdapter

# Canadian statements
canadian_crawler = CanadianAdapter()
canadian_data = canadian_crawler.crawl_and_save("output/canadian_statements.csv")

# Australian statements with country filter
australian_crawler = AustralianAdapter(country="Canada")
australian_data = australian_crawler.crawl_and_save("output/australian_canada.csv")
```

## Extending the Framework

### Adding New Jurisdictions

The framework is designed for easy extension. To add a new jurisdiction:

1. **Create configuration factory** in `configs/new_jurisdiction.py`:
```python
def create_config() -> CrawlerConfig:
    return CrawlerConfig(
        name="New Jurisdiction Register",
        jurisdiction_code="NJ",
        search_url_template="https://new-site.gov/search?page={page}",
        base_url="https://new-site.gov/",
        page_range=(1, 100)
    )
```

2. **Create adapter** in `adapters/new_jurisdiction.py`:
```python
class NewJurisdictionAdapter(BaseCrawler):
    def extract_statement_links(self, soup: BeautifulSoup) -> List[str]:
        # Extract URLs to individual statement pages
        pass
    
    def extract_statement_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        # Extract structured data from statement pages
        pass
```

3. **Register in configuration** (`config.py`):
```python
from .configs.new_jurisdiction import create_config as create_new_config
JURISDICTION_CONFIGS = {
    'new_jurisdiction': create_new_config,
    # ... existing jurisdictions
}
```

4. **Add to CLI** (`main.py`) and exports (`adapters/__init__.py`)

## Why This Architecture is Maintainable and Extensible

### 1. **Single Responsibility Principle**
Each module handles exactly one concern, making the codebase easier to understand, test, and modify.

### 2. **Dependency Injection**
Adapters receive their configuration and dependencies, making them testable and flexible.

### 3. **Abstract Base Classes**
The `BaseCrawler` interface ensures consistent behavior across jurisdictions while allowing site-specific customization.

### 4. **Modular Core Components**
The core crawling logic is separated into focused components that can be reused and tested independently.

### 5. **Configuration-Driven**
Site-specific behavior is controlled through configuration rather than hardcoded values, enabling easy customization.

### 6. **Isolated Jurisdictions**
Each jurisdiction's implementation is completely isolated - changes to Canadian crawling don't affect Australian crawling.

### 7. **Hardcoded Registry**
The jurisdiction registry uses explicit imports rather than dynamic discovery, preventing runtime errors and making the system more predictable.
