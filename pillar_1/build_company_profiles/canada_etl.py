"""
Canadian Modern Slavery ETL Implementation
==========================================

Implementation of the base ETL system for Canadian data sources.

"""

import polars as pl
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging
from datetime import datetime
from enum import Enum

from etl_base import (
    BaseDataReader,
    BaseETL,
    ETLConfig,
    DataQualityReport,
    CanadianStatementColumns,
    CanadianBusinessColumns,
    normalize_company_name,
    log_join_statistics,
    DataLoadException,
    DataJoinException,
)
from column_standardization import ColumnStandardizer

logger = logging.getLogger(__name__)


class CanadianStatementsReader(BaseDataReader):
    """
    Reader for Canadian Modern Slavery Statements.

    Parameters
    ----------
    file_path : str
        Path to the statements CSV file.
    config : ETLConfig
        Configuration object for ETL processing.

    """

    def load_data(self) -> pl.DataFrame:
        """
        Load Canadian statements data.

        Returns
        -------
        pl.DataFrame
            Cleaned statements dataframe.

        Raises
        ------
        DataLoadException
            If data loading fails.

        """
        logger.info(f"Loading Canadian statements: {self.file_path.name}")

        try:
            df = pl.read_csv(
                self.file_path,
                ignore_errors=self.config.ignore_errors,
                infer_schema_length=self.config.infer_schema_length,
            )

            df = self.clean_column_names(df)
            df = self.select_essential_columns(
                df, CanadianStatementColumns.get_all_columns()
            )

            df = self._clean_data_types(df)

            logger.info(f"Loaded {len(df)} statement records")
            return df

        except Exception as e:
            raise DataLoadException(f"Failed to load statements: {e}")

    def _clean_data_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean and standardize data types.

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

        if "organization" in df.columns:
            operations.append(
                pl.col("organization").str.strip_chars().alias("organization")
            )

        if "year" in df.columns:
            operations.append(pl.col("year").cast(pl.Int64, strict=False))

        if "language" in df.columns:
            operations.append(
                pl.col("language")
                .str.to_lowercase()
                .str.strip_chars()
                .alias("language")
            )

        return df.with_columns(operations) if operations else df

    def get_most_recent_statements(self) -> pl.DataFrame:
        """
        Get the most recent statement for each organization.

        Returns
        -------
        pl.DataFrame
            DataFrame with most recent statements per organization.

        """
        logger.info("Processing most recent statements per organization...")

        df = self.load_data()

        df = df.filter(
            pl.col("organization").is_not_null() & (pl.col("organization") != "")
        )

        df = df.sort(["organization", "year"], descending=[False, True])
        df_recent = df.unique(subset=["organization"], keep="first")

        logger.info(f"Unique organizations with statements: {len(df_recent)}")

        self._log_distribution_statistics(df_recent)

        return df_recent

    def _log_distribution_statistics(self, df: pl.DataFrame):
        """
        Log distribution statistics for the dataset.

        Parameters
        ----------
        df : pl.DataFrame
            Dataset to analyze.

        """
        if "year" in df.columns:
            year_dist = df.group_by("year").len().sort("year")
            logger.info("Statement year distribution:")
            for row in year_dist.iter_rows():
                logger.info(f"  {row[0]}: {row[1]} organizations")

        if "language" in df.columns:
            lang_dist = df.group_by("language").len().sort("len", descending=True)
            logger.info("Language distribution:")
            for row in lang_dist.iter_rows():
                logger.info(f"  {row[0]}: {row[1]} organizations")


class CanadianBusinessReader(BaseDataReader):
    """
    Reader for Canadian Business Registry data.

    Parameters
    ----------
    file_path : str
        Path to the business registry CSV file.
    config : ETLConfig
        Configuration object for ETL processing.

    """

    def load_data(self) -> pl.DataFrame:
        """
        Load Canadian business registry data.

        Returns
        -------
        pl.DataFrame
            Cleaned business registry dataframe.

        Raises
        ------
        DataLoadException
            If data loading fails.

        """
        logger.info(f"Loading Canadian business data: {self.file_path.name}")

        try:
            df = pl.read_csv(
                self.file_path,
                ignore_errors=self.config.ignore_errors,
                infer_schema_length=self.config.infer_schema_length,
            )

            df = self.clean_column_names(df)
            df = self.select_essential_columns(
                df, CanadianBusinessColumns.get_all_columns()
            )

            df = self._clean_data_types(df)

            logger.info(f"Loaded {len(df)} business records")
            return df

        except Exception as e:
            raise DataLoadException(f"Failed to load business data: {e}")

    def _clean_data_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean and standardize data types for business data.

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

        if "business_name" in df.columns:
            operations.append(
                pl.col("business_name").str.strip_chars().alias("business_name")
            )

        if "business_id_no" in df.columns:
            operations.append(
                pl.col("business_id_no").str.strip_chars().alias("business_id_no")
            )

        if "prov_terr" in df.columns:
            operations.append(
                pl.col("prov_terr")
                .str.to_uppercase()
                .str.strip_chars()
                .alias("prov_terr")
            )

        return df.with_columns(operations) if operations else df


class CanadianCompanyETL(BaseETL):
    """
    ETL implementation for Canadian Modern Slavery data.

    Parameters
    ----------
    config : ETLConfig
        Configuration object containing file paths and settings.

    """

    def _setup_readers(self):
        """Initialize Canadian data readers."""
        self.statements_reader = CanadianStatementsReader(
            self.config.statements_path, self.config
        )
        self.business_reader = CanadianBusinessReader(
            self.config.registry_path, self.config
        )

    def _create_fuzzy_match_key(self, df: pl.DataFrame, name_col: str) -> pl.DataFrame:
        """
        Create standardized key for fuzzy matching.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.
        name_col : str
            Name of the column containing company names.

        Returns
        -------
        pl.DataFrame
            Dataframe with added match key.

        """
        df = df.with_columns(
            [
                pl.col(name_col)
                .str.to_lowercase()
                .str.replace_all(r"[^\w\s]", "", literal=False)
                .str.replace_all(r"\s+", " ", literal=False)
                .str.strip_chars()
                .str.replace_all(
                    r"\b(inc|ltd|corp|corporation|company|co|limited|ltee|ltda)\b",
                    "",
                    literal=False,
                )
                .str.strip_chars()
                .alias(f"{name_col}_match_key")
            ]
        )
        return df

    def create_company_profiles(self) -> pl.DataFrame:
        """
        Create company profiles by joining statements with business registry data.

        Returns
        -------
        pl.DataFrame
            Joined dataset with company profiles.

        Raises
        ------
        DataJoinException
            If joining fails.

        """
        logger.info("Creating Canadian company profiles...")

        try:
            statements_df = self.statements_reader.get_most_recent_statements()
            business_df = self.business_reader.load_data()

            self._log_duplicate_businesses(business_df)

            logger.info("Attempting direct join on parent_company -> business_name...")
            direct_join = statements_df.join(
                business_df,
                left_on="parent_company",
                right_on="business_name",
                how="left",
                suffix="_biz",
            )

            direct_join = direct_join.unique(subset=["parent_company"], keep="first")
            direct_matches = direct_join.filter(pl.col("idx").is_not_null()).height

            logger.info("Creating fuzzy match keys...")
            statements_fuzzy = self._create_fuzzy_match_key(
                statements_df, "parent_company"
            )
            business_fuzzy = self._create_fuzzy_match_key(business_df, "business_name")

            logger.info("Performing fuzzy join...")
            fuzzy_join = statements_fuzzy.join(
                business_fuzzy,
                left_on="parent_company_match_key",
                right_on="business_name_match_key",
                how="left",
                suffix="_biz",
            )

            fuzzy_join = fuzzy_join.unique(subset=["parent_company"], keep="first")
            fuzzy_matches = fuzzy_join.filter(pl.col("idx").is_not_null()).height

            if fuzzy_matches > direct_matches:
                logger.info("Using fuzzy matching results")
                joined_df = fuzzy_join
                successful_joins = fuzzy_matches
            else:
                logger.info("Using direct matching results")
                joined_df = direct_join
                successful_joins = direct_matches

            self._log_join_results(
                statements_df, direct_matches, fuzzy_matches, successful_joins
            )

            return joined_df

        except Exception as e:
            raise DataJoinException(f"Failed to create company profiles: {e}")

    def _log_duplicate_businesses(self, business_df: pl.DataFrame):
        """
        Log information about duplicate business names.

        Parameters
        ----------
        business_df : pl.DataFrame
            Business registry dataframe.

        """
        duplicates = (
            business_df.group_by("business_name")
            .agg([pl.len().alias("count")])
            .filter(pl.col("count") > 1)
            .sort("count", descending=True)
        )

        if len(duplicates) > 0:
            logger.info(f"Found {len(duplicates)} duplicate business names")
            logger.info("Top duplicate business names:")
            for row in duplicates.head(5).iter_rows():
                logger.info(f"  '{row[0]}': {row[1]} occurrences")

    def _log_join_results(
        self,
        statements_df: pl.DataFrame,
        direct_matches: int,
        fuzzy_matches: int,
        successful_joins: int,
    ):
        """
        Log detailed join results.

        Parameters
        ----------
        statements_df : pl.DataFrame
            Original statements dataframe.
        direct_matches : int
            Number of direct matches.
        fuzzy_matches : int
            Number of fuzzy matches.
        successful_joins : int
            Total successful joins.

        """
        total = len(statements_df)
        with_parent = statements_df.filter(
            pl.col("parent_company").is_not_null()
        ).height

        logger.info("Join Results:")
        logger.info(f"  Total statements: {total:,}")
        logger.info(f"  Statements with parent_company: {with_parent:,}")
        logger.info(f"  Direct matches: {direct_matches:,}")
        logger.info(f"  Fuzzy matches: {fuzzy_matches:,}")
        logger.info(f"  Final successful joins: {successful_joins:,}")

        if with_parent > 0:
            logger.info(
                f"  Join rate (of those with parent): "
                f"{successful_joins/with_parent*100:.1f}%"
            )
        logger.info(f"  Overall join rate: {successful_joins/total*100:.1f}%")

    def add_derived_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add derived metrics specific to Canadian data.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.

        Returns
        -------
        pl.DataFrame
            Enhanced dataframe with derived metrics.

        """
        logger.info("Adding Canadian-specific derived metrics...")

        df = df.with_columns(
            [
                pl.col("pdf_url").is_not_null().alias("HasPDFUrl"),
                pl.col("statement_url").is_not_null().alias("HasStatementUrl"),
                pl.col("parent_company").is_not_null().alias("HasParentCompany"),
                pl.col("subsidiary_companies").is_not_null().alias("HasSubsidiaries"),
                pl.col("status").eq("Active").alias("IsActiveBusiness"),
                pl.col("total_no_employees").is_not_null().alias("HasEmployeeCount"),
                pl.when(pl.col("prov_terr").is_in(["ON", "QC", "BC", "AB"]))
                .then(True)
                .otherwise(False)
                .alias("InMajorProvince"),
                pl.col("derived_NAICS").is_not_null().alias("HasNAICSCode"),
            ]
        )

        df = df.with_columns(
            [
                (
                    pl.col("HasPDFUrl").cast(pl.Int32)
                    + pl.col("HasStatementUrl").cast(pl.Int32)
                    + pl.col("HasParentCompany").cast(pl.Int32)
                    + pl.col("HasNAICSCode").cast(pl.Int32)
                ).alias("DataCompletenessScore")
            ]
        )

        if "year" in df.columns:
            current_year = datetime.now().year
            df = df.with_columns(
                [
                    pl.when(pl.col("year").is_not_null())
                    .then(pl.max_horizontal(0, 5 - (current_year - pl.col("year"))))
                    .otherwise(0)
                    .alias("RecencyScore")
                ]
            )

        return df
    
    def standardize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply Canadian-specific column standardization.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe with original column names.
        
        Returns
        -------
        pl.DataFrame
            Dataframe with standardized column names.
        
        """
        logger.info("Applying Canadian column standardization...")
        return ColumnStandardizer.standardize_dataframe(df, 'canada')

    def generate_data_quality_report(self, df: pl.DataFrame) -> DataQualityReport:
        """
        Generate data quality report for Canadian data.

        Parameters
        ----------
        df : pl.DataFrame
            Final processed dataframe.

        Returns
        -------
        DataQualityReport
            Comprehensive data quality metrics.

        """
        logger.info("Generating Canadian data quality report...")

        unique_orgs = df.select("organization_name").n_unique()

        key_fields = [
            'organization_name',  
            'company_name',
            'incorporation_date',
            'company_status', 
            'annual_revenue',
            'industry_sectors'
        ]

        completeness = self.calculate_data_completeness(df, key_fields)

        geographic_dist = {}
        if "province_state" in df.columns:
            geo_data = (
                df.filter(pl.col("province_state").is_not_null())
                .group_by("province_state")
                .agg([pl.len().alias("count")])
                .sort("count", descending=True)
            )
            geographic_dist = geo_data.to_dicts()

        sector_dist = {}
        if "industry_sector" in df.columns:
            sector_data = (
                df.filter(pl.col("industry_sector").is_not_null())
                .group_by("industry_sector")
                .agg([pl.len().alias("count")])
                .sort("count", descending=True)
                .head(10)
            )
            sector_dist = sector_data.to_dicts()

        quality_metrics = {}
        if "data_completeness_score" in df.columns:
            avg_completeness = df.select(pl.col("data_completeness_score").mean()).item()
            quality_metrics["average_completeness_score"] = round(avg_completeness, 2)

        return DataQualityReport(
            total_records=len(df),
            unique_entities=unique_orgs,
            unique_statements=unique_orgs,
            data_completeness=completeness,
            join_quality_metrics=quality_metrics,
            data_distribution={"geographic": geographic_dist, "sector": sector_dist},
        )
