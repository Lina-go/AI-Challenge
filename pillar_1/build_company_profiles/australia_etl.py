"""
Australian Modern Slavery ETL Implementation
============================================

Implementation of the base ETL system for Australian data sources.

"""

import polars as pl
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set
import logging
from datetime import datetime

from etl_base import (
    BaseDataReader,
    BaseETL,
    ETLConfig,
    DataQualityReport,
    AustralianStatementColumns,
    ASICColumns,
    normalize_company_name,
    log_join_statistics,
    DataLoadException,
    DataJoinException,
)
from column_standardization import ColumnStandardizer

logger = logging.getLogger(__name__)


class AustralianStatementsReader(BaseDataReader):
    """
    Reader for Australian Modern Slavery Statements.

    Parameters
    ----------
    file_path : str
        Path to the statements CSV file.
    config : ETLConfig
        Configuration object for ETL processing.

    """

    def discover_statement_files(self) -> List[Path]:
        """
        Discover all statement CSV files in the directory.

        Returns
        -------
        List[Path]
            List of CSV file paths found.

        Raises
        ------
        FileNotFoundError
            If no statement files are found.

        """
        pattern = "au_*.csv"
        files = list(self.file_path.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No statement files found matching {pattern}")

        logger.info(f"Found {len(files)} statement files: {[f.name for f in files]}")
        return sorted(files)
    
    def load_single_statement_file(self, file_path: Path) -> pl.DataFrame:
        """
        Load and clean a single statement file.

        Parameters
        ----------
        file_path : Path
            Path to the statement CSV file.

        Returns
        -------
        pl.DataFrame
            Cleaned dataframe with selected columns.

        """
        logger.info(f"Loading statement file: {file_path.name}")

        try:
            df = pl.read_csv(
                file_path,
                ignore_errors=self.config.ignore_errors,
                infer_schema_length=self.config.infer_schema_length,
            )

            df = self.clean_column_names(df)
            df = self.select_essential_columns(df, AustralianStatementColumns.get_all_columns())

            df = self._clean_data_types(df)

            logger.info(f"Loaded {len(df)} records from {file_path.name}")
            return df

        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            raise

    def load_data(self) -> pl.DataFrame:
        """
        Load all statement files and concatenate them.

        Returns
        -------
        pl.DataFrame
            Combined dataframe of all statements.

        Raises
        ------
        DataLoadException
            If data loading fails.

        """
        try:
            files = self.discover_statement_files()
            all_dfs = []

            for file_path in files:
                df = self.load_single_statement_file(file_path)
                all_dfs.append(df)

            combined_df = pl.concat(all_dfs, how="vertical_relaxed")
            logger.info(f"Combined dataset: {len(combined_df)} total records")

            return combined_df

        except Exception as e:
            raise DataLoadException(f"Failed to load UK statements: {e}")

    def _clean_data_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean and standardize data types.
        """
        operations = []

        # Limpiar strings
        for col in ["acn", "abn"]:
            if col in df.columns:
                operations.append(
                    pl.col(col).cast(pl.Utf8, strict=False).str.strip_chars().alias(col)
                )

        df = df.with_columns(operations) if operations else df
        
        # Parsear fechas
        df = self.parse_dates(
            df,
            ["period_start", "period_end"],
            date_format="%Y-%m-%d",
        )
        
        return df

    def get_most_recent_statements(self) -> pl.DataFrame:
        """
        Get the most recent statement for each entity.

        Returns
        -------
        pl.DataFrame
            DataFrame with most recent statements per entity.

        """
        logger.info("Processing most recent statements per company...")

        df = self.load_data()

        df = df.with_columns(
            [
                pl.col("reporting_entities")
                .str.strip_chars()
                .str.to_uppercase()
                .alias("reporting_entities_clean"),
                pl.concat_str(
                    [
                        pl.col("idx").cast(pl.Utf8),
                        pl.lit("_"),
                        pl.col("period_start").cast(pl.Utf8),
                        pl.lit("_"),
                        pl.col("period_end").cast(pl.Utf8),
                    ]
                ).alias("statement_unique_id"),
            ]
        )

        df_recent = df.sort(
            ["statement_unique_id", "period_end"], descending=[False, True]
        ).unique(subset=["statement_unique_id"], keep="first")

        logger.info(f"Unique statements found: {len(df_recent)}")

        df_expanded = self._expand_abns_to_entities(df_recent)

        df_expanded = df_expanded.with_columns(
            [pl.col("period_end").dt.year().alias("statement_year")]
        ).drop(["reporting_entities_clean", "statement_unique_id"])

        self._log_expansion_statistics(df_expanded)

        return df_expanded

    def _expand_abns_to_entities(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Expand statements with multiple ABNs into individual entity rows.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe with potentially multiple ABNs per row.

        Returns
        -------
        pl.DataFrame
            Expanded dataframe with one row per entity.

        """
        logger.info("Expanding ABNs to individual entity level...")

        if "abn" not in df.columns:
            return df

        df = (
            df.with_columns(
                [
                    pl.col("abn")
                    .cast(pl.Utf8, strict=False)
                    .str.strip_chars()
                    .str.split(",")
                    .alias("abn_list")
                ]
            )
            .explode("abn_list")
            .with_columns(
                [
                    pl.col("abn_list").str.strip_chars().alias("abn"),
                    pl.col("abn_list").count().over("idx").alias("group_entity_count"),
                    pl.int_range(pl.len()).over("idx").alias("entity_index_in_group"),
                    pl.col("idx").alias("statement_group_id"),
                ]
            )
            .with_columns(
                [(pl.col("entity_index_in_group") == 0).alias("is_primary_entity")]
            )
            .drop("abn_list")
        )

        logger.info(f"Expanded to {df.height} entity-level records")
        return df

    def _log_expansion_statistics(self, df: pl.DataFrame):
        """
        Log statistics about the ABN expansion.

        Parameters
        ----------
        df : pl.DataFrame
            Expanded dataframe.

        """
        stats = df.select(
            [
                pl.len().alias("total_entities"),
                pl.col("abn").is_not_null().sum().alias("entities_with_abn"),
                pl.col("abn").is_null().sum().alias("entities_without_abn"),
            ]
        ).to_dicts()[0]

        logger.info(f"Total entities after expansion: {stats['total_entities']:,}")
        logger.info(f"  With ABN: {stats['entities_with_abn']:,}")
        logger.info(f"  Without ABN: {stats['entities_without_abn']:,}")

        if "statement_year" in df.columns:
            year_dist = df.group_by("statement_year").len().sort("statement_year")
            logger.info("Statement year distribution:")
            for row in year_dist.iter_rows():
                logger.info(f"  {row[0]}: {row[1]:,} entities")


class ASICCompanyReader(BaseDataReader):
    """
    Reader for ASIC company registry data.

    Parameters
    ----------
    file_path : str
        Path to the ASIC CSV file.
    config : ETLConfig
        Configuration object for ETL processing.

    """

    def load_data(self) -> pl.DataFrame:
        """
        Load ASIC company data with deduplication.

        Returns
        -------
        pl.DataFrame
            Cleaned ASIC dataframe.

        Raises
        ------
        DataLoadException
            If data loading fails.

        """
        logger.info(f"Loading ASIC data: {self.file_path.name}")

        try:
            df = pl.read_csv(
                self.file_path,
                separator="\t",
                ignore_errors=self.config.ignore_errors,
                infer_schema_length=self.config.infer_schema_length,
                truncate_ragged_lines=self.config.truncate_ragged_lines,
            )

            df = self.clean_column_names(df)
            df = self.select_essential_columns(df, ASICColumns.get_all_columns())

            df = self._deduplicate_by_current_name(df)
            df = self._clean_data_types(df)
            df = self.parse_dates(
                df,
                [
                    "Date of Registration",
                    "Date of Deregistration",
                    "Current Name Start Date",
                ],
                date_format="%Y-%m-%d",
            )

            logger.info(f"Loaded {len(df)} company records")
            return df

        except Exception as e:
            raise DataLoadException(f"Failed to load ASIC data: {e}")

    def _deduplicate_by_current_name(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Deduplicate companies preferring current name records.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe with potential duplicates.

        Returns
        -------
        pl.DataFrame
            Deduplicated dataframe.

        """
        if "ABN" not in df.columns or "Current Name Indicator" not in df.columns:
            return df

        df = (
            df.with_columns(
                [(pl.col("Current Name Indicator") == "Y").alias("is_current_name")]
            )
            .with_columns(
                [pl.col("is_current_name").sum().over("ABN").alias("y_count")]
            )
            .filter(
                ((pl.col("y_count") > 0) & pl.col("is_current_name"))
                | (pl.col("y_count") == 0)
            )
            .drop(["is_current_name", "y_count"])
        )

        return df

    def _clean_data_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean and standardize data types for ASIC data.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.

        Returns
        -------
        pl.DataFrame
            Dataframe with cleaned data types.

        """
        operations = []

        string_cols = ["ACN", "ABN", "Company Name", "Current Name"]
        for col in string_cols:
            if col in df.columns:
                operations.append(
                    pl.col(col).cast(pl.Utf8, strict=False).str.strip_chars().alias(col)
                )

        return df.with_columns(operations) if operations else df


class AustralianCompanyETL(BaseETL):
    """
    ETL implementation for Australian Modern Slavery data.

    Parameters
    ----------
    config : ETLConfig
        Configuration object containing file paths and settings.

    """

    def _setup_readers(self):
        """Initialize Australian data readers."""
        self.statements_reader = AustralianStatementsReader(
            self.config.statements_path, self.config
        )
        self.asic_reader = ASICCompanyReader(self.config.registry_path, self.config)

    def create_company_profiles(self) -> pl.DataFrame:
        """
        Create company profiles by joining statements with ASIC data.

        Returns
        -------
        pl.DataFrame
            Joined dataset with company profiles.

        Raises
        ------
        DataJoinException
            If joining fails.

        """
        logger.info("Creating entity profiles...")

        try:
            entities_df = self.statements_reader.get_most_recent_statements()
            asic_df = self.asic_reader.load_data()

            entities_df = entities_df.rename({"abn": "ABN"})

            entities_df = self._clean_abn_column(entities_df)
            asic_df = self._clean_abn_column(asic_df)

            entities_df = entities_df.with_columns(
                [
                    pl.col("reporting_entities")
                    .str.strip_chars()
                    .str.to_uppercase()
                    .alias("entity_name_normalized")
                ]
            )

            asic_df = asic_df.with_columns(
                [
                    pl.col("Company Name")
                    .str.strip_chars()
                    .str.to_uppercase()
                    .alias("company_name_normalized")
                ]
            )

            logger.info("Performing ABN-based join...")
            joined_df = entities_df.join(
                asic_df, on="ABN", how="left", suffix="_asic"
            ).with_columns([pl.lit("ABN").alias("join_method")])

            join_stats = log_join_statistics(
                entities_df, joined_df, "ABN", "Company Name"
            )

            joined_df = self._reorder_columns_abn_first(joined_df)

            return joined_df

        except Exception as e:
            raise DataJoinException(f"Failed to create company profiles: {e}")

    def _clean_abn_column(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean ABN column for joining.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.

        Returns
        -------
        pl.DataFrame
            Dataframe with cleaned ABN column.

        """
        if "ABN" not in df.columns:
            return df

        return df.with_columns(
            [
                pl.when(pl.col("ABN").is_not_null() & (pl.col("ABN") != ""))
                .then(pl.col("ABN").cast(pl.Utf8, strict=False).str.strip_chars())
                .otherwise(None)
                .alias("ABN")
            ]
        )

    def _reorder_columns_abn_first(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Reorder columns to have ABN as the first column.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.

        Returns
        -------
        pl.DataFrame
            Dataframe with reordered columns.

        """
        if "ABN" not in df.columns:
            logger.warning("ABN column not found, cannot reorder")
            return df

        columns = list(df.columns)
        new_order = ["ABN"] + [col for col in columns if col != "ABN"]
        return df.select(new_order)

    def add_derived_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add derived metrics specific to Australian data.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.

        Returns
        -------
        pl.DataFrame
            Enhanced dataframe with derived metrics.

        """
        logger.info("Adding derived metrics...")

        basic_operations = [
            (
                (
                    pl.col("period_end").dt.year()
                    - pl.col("Date of Registration").dt.year()
                ).alias("company_age_at_statement")
            ),
            pl.col("Status").eq("Registered").alias("is_active"),
            pl.col("Date of Deregistration").is_not_null().alias("is_deregistered"),
            (pl.col("period_end") - pl.col("period_start"))
            .dt.total_days()
            .alias("statement_period_days"),
            pl.col("Company Name").is_not_null().alias("has_company_data"),
            pl.col("group_entity_count").gt(1).alias("is_multi_entity_statement"),
            pl.col("annual_revenue").is_not_null().alias("has_revenue_data"),
            pl.col("statement_url").is_not_null().alias("has_statement_url"),
            pl.col("pdf_url").is_not_null().alias("has_pdf_url"),
            pl.col("industry_sectors").is_not_null().alias("has_industry_data"),
        ]

        df = df.with_columns(basic_operations)

        derived_operations = [
            pl.when(pl.col("headquartered_countries").is_not_null())
            .then(pl.col("headquartered_countries").str.count_matches(",") + 1)
            .otherwise(0)
            .alias("num_headquarter_countries"),
        ]

        df = df.with_columns(derived_operations)

        final_operations = [
            (
                pl.col("has_revenue_data").cast(pl.Int32)
                + pl.col("has_statement_url").cast(pl.Int32)
                + pl.col("has_pdf_url").cast(pl.Int32)
                + pl.col("has_industry_data").cast(pl.Int32)
            ).alias("data_completeness_score")
        ]

        df = df.with_columns(final_operations)

        return df
    
    def standardize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply Australian-specific column standardization.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe with original column names.
        
        Returns
        -------
        pl.DataFrame
            Dataframe with standardized column names.
        
        """
        logger.info("Applying Australian column standardization...")
        return ColumnStandardizer.standardize_dataframe(df, 'australia')

    def generate_data_quality_report(self, df: pl.DataFrame) -> DataQualityReport:
        """
        Generate data quality report for Australian data.

        Parameters
        ----------
        df : pl.DataFrame
            Final processed dataframe.

        Returns
        -------
        DataQualityReport
            Comprehensive data quality metrics.

        """
        logger.info("Generating data quality report...")

        basic_stats = df.select(
            [
                pl.len().alias("total_records"),
                pl.col("abn").n_unique().alias("unique_entities"),
                pl.col("statement_group_id").n_unique().alias("unique_statements"),
            ]
        ).to_dicts()[0]

        key_fields = [
            'organization_name', 
            'company_name',      
            'incorporation_date',
            'company_status',  
            'annual_revenue',
            'industry_sectors'
        ]

        completeness = self.calculate_data_completeness(df, key_fields)

        join_metrics = {}
        if "join_method" in df.columns:
            join_stats = df.group_by("join_method").len().to_dicts()
            total = sum(stat["len"] for stat in join_stats)

            for stat in join_stats:
                method = stat["join_method"] if stat["join_method"] else "NO_MATCH"
                count = stat["len"]
                join_metrics[f"{method}_joins"] = {
                    "count": count,
                    "percentage": round((count / total) * 100, 1),
                }

        entity_structure_metrics = {}
        if all(
            col in df.columns
            for col in ["statement_group_id", "is_primary_entity", "group_entity_count"]
        ):
            entity_stats = df.select(
                [
                    pl.col("is_primary_entity").sum().alias("primary_entities"),
                    (pl.col("is_primary_entity") == False)
                    .sum()
                    .alias("subsidiary_entities"),
                    (pl.col("group_entity_count") == 1)
                    .sum()
                    .alias("single_entity_statements"),
                    (pl.col("group_entity_count") > 1)
                    .sum()
                    .alias("multi_entity_statements"),
                    (pl.len() / pl.col("statement_group_id").n_unique()).alias(
                        "avg_entities_per_statement"
                    ),
                ]
            ).to_dicts()[0]

            entity_structure_metrics = {
                "entity_analysis": entity_stats,
                "statement_analysis": {
                    "total_statements": basic_stats["unique_statements"],
                    "single_entity_statements": entity_stats[
                        "single_entity_statements"
                    ],
                    "multi_entity_statements": entity_stats["multi_entity_statements"],
                },
            }

        return DataQualityReport(
            total_records=basic_stats["total_records"],
            unique_entities=basic_stats["unique_entities"],
            unique_statements=basic_stats["unique_statements"],
            data_completeness=completeness,
            join_quality_metrics=join_metrics,
            data_distribution={},
            custom_metrics={"entity_structure_metrics": entity_structure_metrics},
        )
