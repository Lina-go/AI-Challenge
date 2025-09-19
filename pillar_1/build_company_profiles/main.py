#!/usr/bin/env python
"""
Modern Slavery ETL System - Main Execution Script
=================================================

Unified execution script for processing Modern Slavery Statements
across multiple jurisdictions (Australia, Canada, UK).

Usage Examples
--------------
Command line:
    python main.py --country all --output-dir output/ --config config.yaml
    python main.py --country australia --output-dir output/ --config config.yaml

Python:
    from main import run_etl
    results = run_etl('australia', config_file='config.yaml')

"""

import argparse
import sys
import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import polars as pl

from etl_base import ETLConfig, DataQualityReport, ETLException
from australia_etl import AustralianCompanyETL
from canada_etl import CanadianCompanyETL
from uk_etl import UKCompanyETL



logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================


class ConfigManager:
    """
    Manages configuration for all ETL pipelines.

    Parameters
    ----------
    config_file : Optional[str]
        Path to YAML configuration file.

    """

    DEFAULT_CONFIG = {
        "australia": {
            "statements_path": "data/australia/statements.csv",
            "registry_path": "data/australia/asic_registry.tsv",
            "output_path": "output/australia_enriched.csv",
        },
        "canada": {
            "statements_path": "data/canada/statements.csv",
            "registry_path": "data/canada/business_registry.csv",
            "output_path": "output/canada_enriched.csv",
        },
        "uk": {
            "statements_path": "data/uk/statements/",
            "registry_path": "data/uk/companies_house.csv",
            "output_path": "output/uk_enriched.csv",
        },
        "settings": {
            "date_format": "%Y-%m-%d",
            "infer_schema_length": 1000,
            "ignore_errors": True,
            "truncate_ragged_lines": True,
        },
    }

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager."""
        self.config = self.DEFAULT_CONFIG.copy()

        if config_file:
            self.load_from_file(config_file)

    def load_from_file(self, config_file: str):
        """
        Load configuration from YAML file.

        Parameters
        ----------
        config_file : str
            Path to YAML configuration file.

        """
        config_path = Path(config_file)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_file}. Using defaults.")
            return

        try:
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
                self.config = self._merge_configs(self.config, file_config)
                logger.info(f"Configuration loaded from: {config_file}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")

    def _merge_configs(self, base: Dict, update: Dict) -> Dict:
        """
        Recursively merge configuration dictionaries.

        Parameters
        ----------
        base : Dict
            Base configuration.
        update : Dict
            Updates to apply.

        Returns
        -------
        Dict
            Merged configuration.

        """
        result = base.copy()
        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def get_etl_config(self, country: str, overrides: Dict = None) -> ETLConfig:
        """
        Get ETL configuration for a specific country.

        Parameters
        ----------
        country : str
            Country code (australia, canada, uk).
        overrides : Dict, optional
            Override specific configuration values.

        Returns
        -------
        ETLConfig
            Configuration object for ETL processing.

        """
        if country not in self.config:
            raise ValueError(f"Unknown country: {country}")

        country_config = self.config[country].copy()
        settings = self.config["settings"].copy()

        if overrides:
            country_config.update(overrides)

        return ETLConfig(
            statements_path=country_config["statements_path"],
            registry_path=country_config["registry_path"],
            output_path=country_config.get("output_path"),
            date_format=settings["date_format"],
            infer_schema_length=settings["infer_schema_length"],
            ignore_errors=settings["ignore_errors"],
            truncate_ragged_lines=settings["truncate_ragged_lines"],
        )

    def save_template(self, output_path: str = "config_template.yaml"):
        """
        Save a template configuration file.

        Parameters
        ----------
        output_path : str
            Path to save template configuration.

        """
        with open(output_path, "w") as f:
            yaml.dump(self.DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Template configuration saved to: {output_path}")


# ============================================================================
# ETL RUNNER
# ============================================================================


class ETLRunner:
    """
    Orchestrates ETL execution across multiple countries.

    Parameters
    ----------
    config_manager : ConfigManager
        Configuration manager instance.

    """

    ETL_CLASSES = {
        "australia": AustralianCompanyETL,
        "canada": CanadianCompanyETL,
        "uk": UKCompanyETL,
    }

    def __init__(self, config_manager: ConfigManager):
        """Initialize ETL runner."""
        self.config_manager = config_manager
        self.results = {}

    def run_country(
        self, country: str, overrides: Dict = None
    ) -> Tuple[pl.DataFrame, DataQualityReport]:
        """
        Run ETL for a specific country.

        Parameters
        ----------
        country : str
            Country code (australia, canada, uk).
        overrides : Dict, optional
            Configuration overrides.

        Returns
        -------
        Tuple[pl.DataFrame, DataQualityReport]
            Final dataset and quality report.

        Raises
        ------
        ETLException
            If ETL processing fails.

        """
        if country not in self.ETL_CLASSES:
            raise ValueError(f"Unsupported country: {country}")

        logger.info(f"\n{'='*80}")
        logger.info(f"Starting ETL for: {country.upper()}")
        logger.info(f"{'='*80}")

        try:
            config = self.config_manager.get_etl_config(country, overrides)
            etl_class = self.ETL_CLASSES[country]
            etl = etl_class(config)

            df, report = etl.run_full_etl()
            self.results[country] = (df, report)

            return df, report

        except Exception as e:
            logger.error(f"ETL failed for {country}: {e}")
            raise ETLException(f"ETL failed for {country}: {e}")

    def run_all(
        self, countries: Optional[List[str]] = None
    ) -> Dict[str, Tuple[pl.DataFrame, DataQualityReport]]:
        """
        Run ETL for multiple countries.

        Parameters
        ----------
        countries : Optional[List[str]]
            List of country codes. If None, runs all available countries.

        Returns
        -------
        Dict[str, Tuple[pl.DataFrame, DataQualityReport]]
            Results for each country.

        """
        if countries is None:
            countries = list(self.ETL_CLASSES.keys())

        results = {}
        failed = []

        for country in countries:
            try:
                df, report = self.run_country(country)
                results[country] = (df, report)
            except Exception as e:
                logger.error(f"Failed to process {country}: {e}")
                failed.append(country)

        if failed:
            logger.warning(f"Failed countries: {', '.join(failed)}")

        self.results = results
        return results

    def generate_summary_report(self, output_file: Optional[str] = None) -> Dict:
        """
        Generate a summary report across all processed countries.

        Parameters
        ----------
        output_file : Optional[str]
            Path to save summary report as JSON.

        Returns
        -------
        Dict
            Summary statistics across all countries.

        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "countries_processed": list(self.results.keys()),
            "total_records": 0,
            "total_unique_entities": 0,
            "country_details": {},
        }

        for country, (df, report) in self.results.items():
            summary["total_records"] += len(df)
            summary["total_unique_entities"] += report.unique_entities

            summary["country_details"][country] = {
                "records": len(df),
                "unique_entities": report.unique_entities,
                "unique_statements": report.unique_statements,
                "average_completeness": self._calculate_avg_completeness(report),
                "join_success_rate": self._extract_join_rate(report),
            }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary report saved to: {output_file}")

        return summary

    def _calculate_avg_completeness(self, report: DataQualityReport) -> float:
        """Calculate average data completeness from report."""
        if not report.data_completeness:
            return 0.0
        return round(
            sum(report.data_completeness.values()) / len(report.data_completeness), 1
        )

    def _extract_join_rate(self, report: DataQualityReport) -> float:
        """Extract join success rate from report."""
        join_metrics = report.join_quality_metrics

        if not join_metrics:
            return 0.0

        # Try different keys used by different countries
        for key in ["ABN_joins", "business_registry_match_rate", "compliance_metrics"]:
            if key in join_metrics:
                if isinstance(join_metrics[key], dict):
                    if "percentage" in join_metrics[key]:
                        return join_metrics[key]["percentage"]
                elif isinstance(join_metrics[key], (int, float)):
                    return float(join_metrics[key])

        return 0.0


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def create_parser():
    """
    Create command line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.

    """
    parser = argparse.ArgumentParser(
        description="Modern Slavery ETL System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --country australia
  python main.py --country all --config config.yaml
  python main.py --interactive
  python main.py --create-config
        """,
    )

    parser.add_argument(
        "--country",
        choices=["australia", "canada", "uk", "all"],
        help='Country to process (or "all" for all countries)',
    )

    parser.add_argument("--config", help="Path to configuration file (YAML)")

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for results (default: output)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a template configuration file",
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate summary report from existing outputs",
    )

    return parser


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def run_etl(
    country: str,
    config_file: Optional[str] = None,
    output_dir: str = "output",
    log_level: str = "INFO",
) -> Dict[str, Any]:
    """
    Run ETL programmatically.

    Parameters
    ----------
    country : str
        Country code or 'all'.
    config_file : Optional[str]
        Path to configuration file.
    output_dir : str
        Output directory for results.
    log_level : str
        Logging level.

    Returns
    -------
    Dict[str, Any]
        ETL results.

    Examples
    --------
    >>> results = run_etl('australia', config_file='config.yaml')
    >>> df, report = results['australia']

    """


    # Initialize
    config_manager = ConfigManager(config_file)
    runner = ETLRunner(config_manager)

    # Run ETL
    if country == "all":
        results = runner.run_all()
    else:
        overrides = {"output_path": str(Path(output_dir) / f"{country}_enriched.csv")}
        df, report = runner.run_country(country, overrides)
        results = {country: (df, report)}

    # Generate summary
    summary_file = Path(output_dir) / "summary_report.json"
    summary = runner.generate_summary_report(str(summary_file))

    return {
        "results": results,
        "summary": summary,
        "output_dir": output_dir
    }


def main():
    """Main entry point for command line execution."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle special modes
    if args.create_config:
        config_manager = ConfigManager()
        config_manager.save_template()
        print("Template configuration file created: config_template.yaml")
        return 0

    # Validate arguments for non-interactive mode
    if not args.country:
        parser.print_help()
        return 1

    # Run ETL
    try:
        results = run_etl(args.country, args.config, args.output_dir, args.log_level)

        print(f"\nETL completed successfully!")
        print(f"Output directory: {results['output_dir']}")

        # Print summary
        summary = results["summary"]
        print(f"\nSummary:")
        print(f"  Countries processed: {', '.join(summary['countries_processed'])}")
        print(f"  Total records: {summary['total_records']:,}")
        print(f"  Total unique entities: {summary['total_unique_entities']:,}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
