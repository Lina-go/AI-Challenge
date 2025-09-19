"""
UK Modern Slavery ETL Implementation
====================================

Implementation of the base ETL system for UK data sources.

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
    UKStatementColumns,
    CompaniesHouseColumns,
    normalize_company_name,
    log_join_statistics,
    DataLoadException,
    DataJoinException,
)
from column_standardization import ColumnStandardizer

logger = logging.getLogger(__name__)



class UKStatementsReader(BaseDataReader):
    """
    Reader for UK Modern Slavery Statements with multi-file support.

    Parameters
    ----------
    file_path : str
        Path to the directory containing statement CSV files.
    config : ETLConfig
        Configuration object for ETL processing.

    """

    def __init__(self, file_path: str, config: ETLConfig):
        """Initialize with directory path for UK statements."""
        super().__init__(file_path, config)
        if not self.file_path.is_dir():
            raise ValueError(f"UK statements path must be a directory: {file_path}")

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
        pattern = "StatementSummaries*.csv"
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
            df = self.select_essential_columns(df, UKStatementColumns.get_all_columns())

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

        if "CompanyNumber" in df.columns:
            operations.append(
                pl.col("CompanyNumber").str.strip_chars().alias("CompanyNumber")
            )

        if "StatementYear" in df.columns:
            operations.append(pl.col("StatementYear").cast(pl.Int64, strict=False))

        return df.with_columns(operations) if operations else df

    def get_most_recent_statements(self) -> pl.DataFrame:
        """
        Get the most recent statement for each company.

        Returns
        -------
        pl.DataFrame
            DataFrame with most recent statements per company.

        """
        logger.info("Processing most recent statements per company...")

        df = self.load_data()

        df = df.filter(
            pl.col("CompanyNumber").is_not_null() & (pl.col("CompanyNumber") != "")
        )

        df = df.sort(["CompanyNumber", "StatementYear"], descending=[False, True])
        df_recent = df.unique(subset=["CompanyNumber"], keep="first")

        logger.info(f"Unique companies with statements: {len(df_recent)}")

        self._log_year_distribution(df_recent)

        return df_recent

    def _log_year_distribution(self, df: pl.DataFrame):
        """
        Log statement year distribution.

        Parameters
        ----------
        df : pl.DataFrame
            Dataset to analyze.

        """
        if "StatementYear" in df.columns:
            year_dist = df.group_by("StatementYear").len().sort("StatementYear")
            logger.info("Statement year distribution:")
            for row in year_dist.iter_rows():
                logger.info(f"  {row[0]}: {row[1]} companies")


class CompaniesHouseReader(BaseDataReader):
    """
    Reader for Companies House bulk data.

    Parameters
    ----------
    file_path : str
        Path to the Companies House CSV file.
    config : ETLConfig
        Configuration object for ETL processing.

    """

    def load_data(self) -> pl.DataFrame:
        """
        Load Companies House data.

        Returns
        -------
        pl.DataFrame
            Cleaned Companies House dataframe.

        Raises
        ------
        DataLoadException
            If data loading fails.

        """
        logger.info(f"Loading Companies House data: {self.file_path.name}")

        try:
            df = pl.read_csv(
                self.file_path,
                ignore_errors=self.config.ignore_errors,
                infer_schema_length=self.config.infer_schema_length,
            )

            df = self._handle_column_space_prefix(df)
            df = self.clean_column_names(df)
            df = self.select_essential_columns(
                df, CompaniesHouseColumns.get_all_columns()
            )

            df = self._clean_data_types(df)
            df = self._parse_uk_dates(df)

            logger.info(f"Loaded {len(df)} company records")
            return df

        except Exception as e:
            raise DataLoadException(f"Failed to load Companies House data: {e}")

    def _handle_column_space_prefix(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Handle the space prefix issue in some column names.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.

        Returns
        -------
        pl.DataFrame
            Dataframe with fixed column names.

        """
        column_mapping = {}
        for col in df.columns:
            if col.startswith(" "):
                column_mapping[col] = col[1:]

        if column_mapping:
            df = df.rename(column_mapping)
            logger.info(f"Fixed {len(column_mapping)} columns with space prefix")

        return df

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

        if "CompanyNumber" in df.columns:
            operations.append(
                pl.col("CompanyNumber").str.strip_chars().alias("CompanyNumber")
            )

        numeric_cols = ["Mortgages.NumMortCharges", "Mortgages.NumMortOutstanding"]
        for col in numeric_cols:
            if col in df.columns:
                operations.append(pl.col(col).cast(pl.Int64, strict=False).alias(col))

        return df.with_columns(operations) if operations else df

    def _parse_uk_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Parse UK format dates (DD/MM/YYYY).

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.

        Returns
        -------
        pl.DataFrame
            Dataframe with parsed dates.

        """
        date_cols = [
            "DissolutionDate",
            "IncorporationDate",
            "Accounts.NextDueDate",
            "Accounts.LastMadeUpDate",
        ]

        operations = []
        for col in date_cols:
            if col in df.columns:
                operations.append(
                    pl.col(col)
                    .str.strptime(pl.Date, format="%d/%m/%Y", strict=False)
                    .alias(col)
                )

        return df.with_columns(operations) if operations else df


class UKCompanyETL(BaseETL):
    """
    ETL implementation for UK Modern Slavery data.

    Parameters
    ----------
    config : ETLConfig
        Configuration object containing file paths and settings.

    """

    def _setup_readers(self):
        """Initialize UK data readers."""
        self.statements_reader = UKStatementsReader(
            self.config.statements_path, self.config
        )
        self.companies_house_reader = CompaniesHouseReader(
            self.config.registry_path, self.config
        )

    def create_company_profiles(self) -> pl.DataFrame:
        """
        Create company profiles by joining statements with Companies House data.

        Returns
        -------
        pl.DataFrame
            Joined dataset with company profiles.

        Raises
        ------
        DataJoinException
            If joining fails.

        """
        logger.info("Creating UK company profiles...")

        try:
            statements_df = self.statements_reader.get_most_recent_statements()
            ch_df = self.companies_house_reader.load_data()

            logger.info("Joining statements with Companies House data...")

            joined_df = statements_df.join(
                ch_df, on="CompanyNumber", how="left", suffix="_ch"
            )

            join_stats = log_join_statistics(
                statements_df, joined_df, "CompanyNumber", "CompanyName"
            )

            return joined_df

        except Exception as e:
            raise DataJoinException(f"Failed to create company profiles: {e}")

    def add_derived_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add derived metrics specific to UK data.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.

        Returns
        -------
        pl.DataFrame
            Enhanced dataframe with derived metrics.

        """
        logger.info("Adding UK-specific derived metrics...")

        df = df.with_columns(
            [
                (
                    (
                        pl.col("StatementYear") - pl.col("IncorporationDate").dt.year()
                    ).alias("CompanyAgeAtStatement")
                ),
                (
                    pl.col("Risk1").is_not_null().cast(pl.Int32)
                    + pl.col("Risk2").is_not_null().cast(pl.Int32)
                    + pl.col("Risk3").is_not_null().cast(pl.Int32)
                ).alias("RiskAssessmentCompleteness"),
                pl.col("StatementIncludesOrgStructure")
                .eq("Yes")
                .alias("HasOrgStructure"),
                pl.col("StatementIncludesPolicies").eq("Yes").alias("HasPolicies"),
                pl.col("StatementIncludesRisksAssessment")
                .eq("Yes")
                .alias("HasRiskAssessment"),
                pl.col("StatementIncludesDueDiligence")
                .eq("Yes")
                .alias("HasDueDiligence"),
                pl.col("StatementIncludesTraining").eq("Yes").alias("HasTraining"),
                pl.col("StatementIncludesGoals").eq("Yes").alias("HasGoals"),
                pl.col("CompanyStatus").eq("Active").alias("IsActive"),
                pl.col("DissolutionDate").is_not_null().alias("IsDissolved"),
                (pl.col("Mortgages.NumMortCharges").fill_null(0) > 0).alias(
                    "HasMortgageCharges"
                ),
                pl.concat_str(
                    [
                        pl.col("RegAddress.AddressLine1"),
                        pl.col("RegAddress.AddressLine2"),
                        pl.col("RegAddress.PostTown"),
                        pl.col("RegAddress.County"),
                        pl.col("RegAddress.PostCode"),
                    ],
                    separator=", ",
                )
                .str.replace(", ,", ",")
                .str.strip_chars(", ")
                .alias("FullRegisteredAddress"),
            ]
        )

        df = df.with_columns(
            [
                (
                    pl.col("HasOrgStructure").cast(pl.Int32)
                    + pl.col("HasPolicies").cast(pl.Int32)
                    + pl.col("HasRiskAssessment").cast(pl.Int32)
                    + pl.col("HasDueDiligence").cast(pl.Int32)
                    + pl.col("HasTraining").cast(pl.Int32)
                    + pl.col("HasGoals").cast(pl.Int32)
                ).alias("ComplianceScore")
            ]
        )

        df = self._categorize_company_size(df)
        df = self._add_risk_flags(df)

        return df

    def _categorize_company_size(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Categorize companies by size based on turnover.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.

        Returns
        -------
        pl.DataFrame
            Dataframe with company size categories.

        """
        if "Turnover" not in df.columns:
            return df

        df = df.with_columns(
            [
                pl.when(pl.col("Turnover").str.contains("0-25"))
                .then(pl.lit("Micro"))
                .when(pl.col("Turnover").str.contains("25-50"))
                .then(pl.lit("Small"))
                .when(pl.col("Turnover").str.contains("50-250"))
                .then(pl.lit("Medium"))
                .when(pl.col("Turnover").str.contains("250"))
                .then(pl.lit("Large"))
                .otherwise(pl.lit("Unknown"))
                .alias("CompanySize")
            ]
        )

        return df

    def _add_risk_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add risk-related flags based on statement content.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.

        Returns
        -------
        pl.DataFrame
            Dataframe with risk flags.

        """
        risk_cols = []

        for i in range(1, 4):
            risk_col = f"Risk{i}"
            if risk_col in df.columns:
                risk_cols.append(pl.col(risk_col).is_not_null().alias(f"HasRisk{i}"))

        if risk_cols:
            df = df.with_columns(risk_cols)

        return df
    
    def standardize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply UK-specific column standardization.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe with original column names.
        
        Returns
        -------
        pl.DataFrame
            Dataframe with standardized column names.
        
        """
        logger.info("Applying UK column standardization...")
        return ColumnStandardizer.standardize_dataframe(df, 'uk')

    def generate_data_quality_report(self, df: pl.DataFrame) -> DataQualityReport:
        """
        Generate data quality report for UK data.

        Parameters
        ----------
        df : pl.DataFrame
            Final processed dataframe.

        Returns
        -------
        DataQualityReport
            Comprehensive data quality metrics.

        """
        logger.info("Generating UK data quality report...")

        unique_companies = df.select("company_number").n_unique()

        key_fields = [
            'organization_name',  
            'company_name',
            'incorporation_date',
            'company_status', 
            'annual_revenue',
            'industry_sectors'
        ]

        completeness = self.calculate_data_completeness(df, key_fields)

        compliance_metrics = {}
        if "compliance_score" in df.columns:
            compliance_dist = (
                df.group_by("compliance_score").len().sort("compliance_score")
            )
            avg_score = df.select(pl.col("compliance_score").mean()).item()

            compliance_metrics = {
                "average_score": round(avg_score, 2) if avg_score else 0,
                "score_distribution": compliance_dist.to_dicts()
            }

        company_status_dist = {}
        if "company_status" in df.columns:
            status_data = (
                df.group_by("company_status").len().sort("len", descending=True)
            )
            company_status_dist = status_data.to_dicts()

        sector_dist = {}
        if "business_sector" in df.columns:
            sector_data = (
                df.filter(pl.col("business_sector").is_not_null())
                .group_by("business_sector")
                .len()
                .sort("len", descending=True)
                .head(10)
            )
            sector_dist = sector_data.to_dicts()

        return DataQualityReport(
            total_records=len(df),
            unique_entities=unique_companies,
            unique_statements=unique_companies,
            data_completeness=completeness,
            join_quality_metrics={"compliance_metrics": compliance_metrics},
            data_distribution={
                "company_status": company_status_dist,
                "sector_types": sector_dist,
            },
        )
