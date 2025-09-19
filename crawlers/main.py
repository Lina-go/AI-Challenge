"""
Main entry point for the modular crawler framework.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

from .config import get_config, list_available_jurisdictions
from .adapters import CanadianAdapter, AustralianAdapter
from .utils import setup_crawler_logging

logger = logging.getLogger(__name__)


def create_crawler(jurisdiction: str, country: str = None):
    """
    Create a crawler instance for the specified jurisdiction.

    Args:
        jurisdiction: Name of the jurisdiction
        country: Optional country filter (for Australian jurisdiction)

    Returns:
        Crawler instance

    Raises:
        ValueError: If jurisdiction is not supported
    """
    jurisdiction_lower = jurisdiction.lower()

    crawler_map = {
        "canadian": CanadianAdapter,
        "australian": AustralianAdapter,
    }

    if jurisdiction_lower not in crawler_map:
        available = ", ".join(crawler_map.keys())
        raise ValueError(
            f"Unsupported jurisdiction: {jurisdiction}. Available: {available}"
        )

    crawler_class = crawler_map[jurisdiction_lower]
    
    # Handle jurisdiction-specific parameters
    if jurisdiction_lower == "australian":
        return crawler_class(country=country)
    elif jurisdiction_lower == "canadian":
        config = get_config(jurisdiction)
        crawler = crawler_class(config)
        # Store country filter for later use
        if hasattr(crawler, '_country_filter'):
            crawler._country_filter = country
        else:
            setattr(crawler, '_country_filter', country)
        return crawler
    else:
        config = get_config(jurisdiction)
        return crawler_class(config)


def crawl_jurisdiction(
    jurisdiction: str,
    output_dir: str = "output",
    file_format: str = "csv",
    log_level: str = "INFO",
    country: str = None,
) -> bool:
    """
    Crawl statements for a specific jurisdiction.

    Args:
        jurisdiction: Name of the jurisdiction to crawl
        output_dir: Output directory for results
        file_format: Output file format ("csv" or "json")
        log_level: Logging level

    Returns:
        True if crawl was successful, False otherwise
    """
    setup_crawler_logging(log_level)

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing crawler for {jurisdiction}")
        
        # Handle jurisdiction-specific output directories
        jurisdiction_lower = jurisdiction.lower()
        if jurisdiction_lower == "canadian":
            jurisdiction_output_dir = output_path / "canadian"
        elif jurisdiction_lower == "australian":
            jurisdiction_output_dir = output_path / "australian"
        else:
            jurisdiction_output_dir = output_path / jurisdiction_lower
        
        jurisdiction_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle Australian multi-country crawling
        if jurisdiction_lower == "australian" and country == "all":
            logger.info("🌍 Multi-country mode: crawling all available countries")
            crawler = create_crawler(jurisdiction)
            results = crawler.crawl_all_countries(str(jurisdiction_output_dir), file_format)
            
            total_statements = sum(len(data) for data in results.values())
            successful_countries = sum(1 for data in results.values() if len(data) > 0)
            
            if total_statements > 0:
                logger.info(f"✅ Multi-country crawl completed: {total_statements:,} statements from {successful_countries} countries")
                return True
            else:
                logger.error("❌ Multi-country crawl failed: no data retrieved")
                return False
        
        # Single country/jurisdiction crawling
        if country and country != "all":
            logger.info(f"Filtering by country: {country}")
        
        crawler = create_crawler(jurisdiction, country=country if country != "all" else None)
        
        jurisdiction_code = crawler.config.jurisdiction_code.lower()
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        if country and country != "all":
            country_code = country.replace(' ', '_').replace('/', '_').lower()
            filename = f"{jurisdiction_code}_{country_code}_{timestamp}.{file_format}"
        else:
            filename = f"{jurisdiction_code}_statements_{timestamp}.{file_format}"
            
        output_file = jurisdiction_output_dir / filename

        logger.info(f"Starting crawl for {crawler.config.name}")
        
        # Handle Canadian country filtering differently (post-processing)
        if jurisdiction_lower == "canadian" and hasattr(crawler, '_country_filter') and crawler._country_filter:
            data = crawler.crawl_and_save_with_country_filter(
                str(output_file), file_format, crawler._country_filter
            )
        else:
            data = crawler.crawl_and_save(
                str(output_file), file_format
            )

        if data:
            logger.info(
                f"✅ Crawl completed successfully: {len(data)} statements saved to {output_file}"
            )
            return True
        else:
            logger.warning("⚠️ Crawl completed but no data was extracted")
            return False

    except Exception as e:
        logger.error(f"❌ Crawl failed: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Modular Modern Slavery Statement Crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl Canadian statements (saves to output/canadian/)
  python -m crawlers.main canadian --output-dir ./output
  
  # Crawl Canadian statements filtered by detected country
  python -m crawlers.main canadian --country "Netherlands"
  
  # Crawl Australian statements for all countries (saves to output/australian/)
  python -m crawlers.main australian --country all
  
  # Crawl Australian statements filtered by single country
  python -m crawlers.main australian --country "United States of America"
  
  # List available countries for any jurisdiction
  python -m crawlers.main --list-countries canadian
  python -m crawlers.main --list-countries australian
  
  # Crawl with JSON output
  python -m crawlers.main canadian --output-dir ./results --format json
  
  # Crawl with debug logging
  python -m crawlers.main canadian --log-level DEBUG
        """,
    )

    parser.add_argument(
        "jurisdiction",
        help=f"Target jurisdiction. Available: {', '.join(list_available_jurisdictions())}",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Output directory for results (default: output)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)",
    )

    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    
    parser.add_argument(
        "--country",
        "-c",
        help="Filter by country. Australian: exact filter, Canadian: post-processing filter based on company names. Example: 'United States of America'. Use 'all' for Australian multi-country crawl.",
    )
    
    parser.add_argument(
        "--list-countries",
        action="store_true",
        help="List available countries for Australian and Canadian jurisdictions and exit",
    )

    args = parser.parse_args()

    # Handle --list-countries flag
    if args.list_countries:
        if args.jurisdiction and args.jurisdiction.lower() == "australian":
            from .adapters.australian import AustralianAdapter
            countries = AustralianAdapter.get_country_info()
            print("Available countries for Australian jurisdiction:")
            print("=" * 50)
            for country, count in countries:
                print(f"  {country:<30} ({count:,} statements)")
            print(f"\nTotal: {len(countries)} countries")
        elif args.jurisdiction and args.jurisdiction.lower() == "canadian":
            from .adapters.canadian import CanadianAdapter
            countries = CanadianAdapter.get_available_countries()
            print("Available countries for Canadian jurisdiction:")
            print("=" * 50)
            print("Note: Countries are detected based on company name patterns")
            print("and may not reflect exact headquarters locations.\n")
            for country in countries:
                print(f"  {country}")
            print(f"\nTotal: {len(countries)} detectable countries")
        else:
            print("--list-countries is available for Australian and Canadian jurisdictions")
        sys.exit(0)

    try:
        success = crawl_jurisdiction(
            args.jurisdiction, args.output_dir, args.format, args.log_level, args.country
        )
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Crawl interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
