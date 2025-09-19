"""
Jurisdiction-specific configurations for the crawler framework.

Each jurisdiction has its own Python file with a create_config() function
that returns a CrawlerConfig instance.

Example structure:
    configs/
    ├── __init__.py          # This file
    └── canadian.py          # Canadian configuration

To add a new jurisdiction:
1. Create a new Python file named after the jurisdiction (e.g., british.py)
2. Implement a create_config() function that returns a CrawlerConfig
3. Import and register it in the parent config.py module
"""
