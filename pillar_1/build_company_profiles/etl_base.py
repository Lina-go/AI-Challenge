"""
Modern Slavery ETL System - Core Components
===========================================

A modular ETL system for processing Modern Slavery Statements
and enriching them with company registry data.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set, Any
import logging
import polars as pl
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONFIGURATION
# ============================================================================


class AustralianStatementColumns(Enum):
    """Column definitions for Australian Modern Slavery Statements."""

    IDX = "idx"
    PERIOD_START = "period_start"
    PERIOD_END = "period_end"
    STATEMENT_TYPE = "statement_type"
    HEADQUARTERED_COUNTRIES = "headquartered_countries"
    ANNUAL_REVENUE = "annual_revenue"
    REPORTING_ENTITIES = "reporting_entities"
    ABN = "abn"
    STATEMENT_URL = "statement_url"
    INDUSTRY_SECTORS = "industry_sectors"
    RELATED_STATEMENTS = "related_statements"
    JURISDICTION = "jurisdiction"
    SOURCE_SYSTEM = "source_system"
    COUNTRY_FILTER = "country_filter"
    PDF_URL = "pdf_url"
    INCLUDED_ENTITIES = "included_entities"

    @classmethod
    def get_all_columns(cls) -> List[str]:
        """Get all column names as a list."""
        return [col.value for col in cls]


class ASICColumns(Enum):
    """Column definitions for ASIC company registry."""

    COMPANY_NAME = "Company Name"
    ACN = "ACN"
    TYPE = "Type"
    CLASS = "Class"
    SUB_CLASS = "Sub Class"
    STATUS = "Status"
    DATE_OF_REGISTRATION = "Date of Registration"
    DATE_OF_DEREGISTRATION = "Date of Deregistration"
    PREVIOUS_STATE_REG = "Previous State of Registration"
    STATE_REG_NUMBER = "State Registration number"
    MODIFIED_SINCE_REPORT = "Modified since last report"
    CURRENT_NAME_INDICATOR = "Current Name Indicator"
    ABN = "ABN"
    CURRENT_NAME = "Current Name"
    CURRENT_NAME_START_DATE = "Current Name Start Date"

    @classmethod
    def get_all_columns(cls) -> List[str]:
        """Get all column names as a list."""
        return [col.value for col in cls]


class CanadianStatementColumns(Enum):
    """Column definitions for Canadian Modern Slavery Statements."""

    TITLE = "title"
    STATEMENT_NAME = "statement_name"
    PDF_URL = "pdf_url"
    STATEMENT_URL = "statement_url"
    YEAR = "year"
    ORGANIZATION = "organization"
    LANGUAGE = "language"
    FILE_TYPE = "file_type"
    REPORTING_LAW = "reporting_law"
    STATEMENT_TYPE = "statement_type"
    LEGAL_FRAMEWORK = "legal_framework"
    INDUSTRY_SECTOR = "industry_sector"
    PARENT_COMPANY = "parent_company"
    SUBSIDIARY_COMPANIES = "subsidiary_companies"
    COMPANY_COUNTRY = "company_country"
    DOCUMENT_TYPE = "document_type"
    SOURCE_URL = "source_url"
    JURISDICTION = "jurisdiction"
    INDUSTRY_SECTORS_ADDITIONAL = "industry_sectors_additional"
    INDUSTRY_SUBSECTOR = "industry_subsector"
    INDUSTRY_SUBSECTORS_ADDITIONAL = "industry_subsectors_additional"

    @classmethod
    def get_all_columns(cls) -> List[str]:
        """Get all column names as a list."""
        return [col.value for col in cls]
    
class CanadianBusinessColumns(Enum):
    """Column definitions for Canadian Business Registry."""

    IDX = "idx"
    BUSINESS_NAME = "business_name"
    ALT_BUSINESS_NAME = "alt_business_name"
    BUSINESS_SECTOR = "business_sector"
    BUSINESS_SUBSECTOR = "business_subsector"
    BUSINESS_DESCRIPTION = "business_description"
    BUSINESS_ID_NO = "business_id_no"
    LICENCE_NUMBER = "licence_number"
    LICENCE_TYPE = "licence_type"
    DERIVED_NAICS = "derived_NAICS"
    SOURCE_NAICS_PRIMARY = "source_NAICS_primary"
    SOURCE_NAICS_SECONDARY = "source_NAICS_secondary"
    NAICS_DESCR = "NAICS_descr"
    NAICS_DESCR2 = "NAICS_descr2"
    FULL_ADDRESS = "full_address"
    CITY = "city"
    PROV_TERR = "prov_terr"
    TOTAL_NO_EMPLOYEES = "total_no_employees"
    STATUS = "status"
    PROVIDER = "provider"

    @classmethod
    def get_all_columns(cls) -> List[str]:
        """Get all column names as a list."""
        return [col.value for col in cls]



class UKStatementColumns(Enum):
    """Column definitions for UK Modern Slavery Statements."""

    STATEMENT_YEAR = "StatementYear"
    ORGANISATION_NAME = "OrganisationName"
    ADDRESS = "Address"
    SECTOR_TYPE = "SectorType"
    COMPANY_NUMBER = "CompanyNumber"
    LAST_UPDATED = "LastUpdated"
    GROUP_SUBMISSION = "GroupSubmission"
    PARENT_NAME = "ParentName"
    STATEMENT_URL = "StatementURL"
    STATEMENT_START_DATE = "StatementStartDate"
    STATEMENT_END_DATE = "StatementEndDate"
    APPROVING_PERSON = "ApprovingPerson"
    DATE_APPROVED = "DateApproved"
    INCLUDES_ORG_STRUCTURE = "StatementIncludesOrgStructure"
    INCLUDES_POLICIES = "StatementIncludesPolicies"
    INCLUDES_RISKS = "StatementIncludesRisksAssessment"
    INCLUDES_DUE_DILIGENCE = "StatementIncludesDueDiligence"
    INCLUDES_TRAINING = "StatementIncludesTraining"
    INCLUDES_GOALS = "StatementIncludesGoals"
    ORGANISATION_SECTORS = "OrganisationSectors"
    TURNOVER = "Turnover"
    YEARS_PRODUCING = "YearsProducingStatements"
    RISK1 = "Risk1"
    RISK1_AREA = "Risk1Area"
    RISK1_LOCATION = "Risk1Location"
    RISK1_MITIGATION = "Risk1Mitigation"
    RISK2 = "Risk2"
    RISK2_AREA = "Risk2Area"
    RISK2_LOCATION = "Risk2Location"
    RISK2_MITIGATION = "Risk2Mitigation"
    RISK3 = "Risk3"
    RISK3_AREA = "Risk3Area"
    RISK3_LOCATION = "Risk3Location"
    RISK3_MITIGATION = "Risk3Mitigation"
    DEMONSTRATE_PROGRESS = "DemonstrateProgress"
    PDF_URL = "PDFURL"

    @classmethod
    def get_all_columns(cls) -> List[str]:
        """Get all column names as a list."""
        return [col.value for col in cls]


class CompaniesHouseColumns(Enum):
    """Column definitions for Companies House registry."""

    COMPANY_NAME = "CompanyName"
    COMPANY_NUMBER = "CompanyNumber"
    REG_ADDRESS_LINE1 = "RegAddress.AddressLine1"
    REG_ADDRESS_LINE2 = "RegAddress.AddressLine2"
    REG_ADDRESS_TOWN = "RegAddress.PostTown"
    REG_ADDRESS_COUNTY = "RegAddress.County"
    REG_ADDRESS_COUNTRY = "RegAddress.Country"
    REG_ADDRESS_POSTCODE = "RegAddress.PostCode"
    COMPANY_CATEGORY = "CompanyCategory"
    COMPANY_STATUS = "CompanyStatus"
    COUNTRY_OF_ORIGIN = "CountryOfOrigin"
    DISSOLUTION_DATE = "DissolutionDate"
    INCORPORATION_DATE = "IncorporationDate"
    ACCOUNTS_NEXT_DUE = "Accounts.NextDueDate"
    ACCOUNTS_LAST_MADE = "Accounts.LastMadeUpDate"
    ACCOUNTS_CATEGORY = "Accounts.AccountCategory"
    MORTGAGES_CHARGES = "Mortgages.NumMortCharges"
    MORTGAGES_OUTSTANDING = "Mortgages.NumMortOutstanding"
    SIC_CODE_1 = "SICCode.SicText_1"
    SIC_CODE_2 = "SICCode.SicText_2"
    SIC_CODE_3 = "SICCode.SicText_3"
    SIC_CODE_4 = "SICCode.SicText_4"
    PREVIOUS_NAME_1 = "PreviousName_1.CompanyName"
    PREVIOUS_NAME_2 = "PreviousName_2.CompanyName"
    URI = "URI"

    @classmethod
    def get_all_columns(cls) -> List[str]:
        """Get all column names as a list."""
        return [col.value for col in cls]

# ============================================================================
# DATA CLASSES FOR CONFIGURATION
# ============================================================================


@dataclass
class ETLConfig:
    """Configuration for ETL processing."""

    statements_path: str
    registry_path: str
    output_path: Optional[str] = None
    date_format: str = "%Y-%m-%d"
    infer_schema_length: int = 1000
    ignore_errors: bool = True
    truncate_ragged_lines: bool = True


@dataclass
class DataQualityReport:
    """Standardized data quality report structure."""

    total_records: int
    unique_entities: int
    unique_statements: int
    data_completeness: Dict[str, float]
    join_quality_metrics: Dict[str, Any]
    data_distribution: Dict[str, Any]
    custom_metrics: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        """Convert report to dictionary format."""
        return {
            "total_records": self.total_records,
            "unique_entities": self.unique_entities,
            "unique_statements": self.unique_statements,
            "data_completeness": self.data_completeness,
            "join_quality_metrics": self.join_quality_metrics,
            "data_distribution": self.data_distribution,
            "custom_metrics": self.custom_metrics or {},
        }


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class ETLException(Exception):
    """Base exception for ETL operations."""

    pass


class DataLoadException(ETLException):
    """Exception raised when data loading fails."""

    pass


class DataJoinException(ETLException):
    """Exception raised when data joining fails."""

    pass


# ============================================================================
# BASE CLASSES
# ============================================================================


class BaseDataReader(ABC):
    """
    Abstract base class for data readers.

    Parameters
    ----------
    file_path : str
        Path to the data file or directory.
    config : ETLConfig
        Configuration object for ETL processing.

    """

    def __init__(self, file_path: str, config: ETLConfig):
        self.file_path = Path(file_path)
        self.config = config
        self._validate_path()

    def _validate_path(self):
        """
        Validate that the file or directory exists.

        Raises
        ------
        FileNotFoundError
            If the specified path does not exist.

        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Path not found: {self.file_path}")

    @abstractmethod
    def load_data(self) -> pl.DataFrame:
        """
        Load data from the source.

        Returns
        -------
        pl.DataFrame
            Loaded and cleaned dataframe.

        """
        pass

    def clean_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean column names by removing extra whitespace.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe with potentially dirty column names.

        Returns
        -------
        pl.DataFrame
            Dataframe with cleaned column names.

        """
        return df.rename({col: col.strip() for col in df.columns})

    def select_essential_columns(
        self, df: pl.DataFrame, essential_columns: List[str]
    ) -> pl.DataFrame:
        """
        Select only essential columns that exist in the dataframe.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.
        essential_columns : List[str]
            List of column names to select.

        Returns
        -------
        pl.DataFrame
            Dataframe with selected columns.

        """
        available = [col for col in essential_columns if col in df.columns]
        missing = [col for col in essential_columns if col not in df.columns]

        if missing:
            logger.warning(f"Missing columns: {missing}")

        return df.select(available)

    def parse_dates(
        self, df: pl.DataFrame, date_columns: List[str], date_format: str = None
    ) -> pl.DataFrame:
        """
        Parse date columns to date type.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.
        date_columns : List[str]
            List of column names containing dates.
        date_format : str, optional
            Date format string. Uses config default if not provided.

        Returns
        -------
        pl.DataFrame
            Dataframe with parsed date columns.

        """
        date_format = date_format or self.config.date_format
        operations = []

        for col in date_columns:
            if col in df.columns:
                operations.append(
                    pl.col(col)
                    .str.strptime(pl.Date, format=date_format, strict=False)
                    .alias(col)
                )

        return df.with_columns(operations) if operations else df


class BaseETL(ABC):
    """
    Abstract base class for ETL operations.

    Parameters
    ----------
    config : ETLConfig
        Configuration object for ETL processing.

    """

    def __init__(self, config: ETLConfig):
        self.config = config
        self._setup_readers()

    @abstractmethod
    def _setup_readers(self):
        """Initialize data readers."""
        pass

    @abstractmethod
    def create_company_profiles(self) -> pl.DataFrame:
        """
        Create company profiles by joining data sources.

        Returns
        -------
        pl.DataFrame
            Joined dataset with company profiles.

        """
        pass

    @abstractmethod
    def add_derived_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add derived metrics to the dataset.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.

        Returns
        -------
        pl.DataFrame
            Enhanced dataframe with derived metrics.

        """
        pass

    def calculate_data_completeness(
        self, df: pl.DataFrame, key_fields: List[str]
    ) -> Dict[str, float]:
        """
        Calculate data completeness percentages for key fields.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.
        key_fields : List[str]
            List of field names to check.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping field names to completeness percentages.

        """
        completeness = {}
        total = len(df)

        for field in key_fields:
            if field in df.columns:
                non_null = df.filter(pl.col(field).is_not_null()).height
                completeness[field] = round((non_null / total) * 100, 1)

        return completeness

    @abstractmethod
    def generate_data_quality_report(self, df: pl.DataFrame) -> DataQualityReport:
        """
        Generate a comprehensive data quality report.

        Parameters
        ----------
        df : pl.DataFrame
            Final processed dataframe.

        Returns
        -------
        DataQualityReport
            Data quality metrics report.

        """
        pass

    def standardize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Standardize column names. Override in subclasses to implement.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.
        
        Returns
        -------
        pl.DataFrame
            Dataframe with standardized columns.
        
        Notes
        -----
        Default implementation returns dataframe unchanged.
        Override this method in country-specific implementations.
        
        """
        return df 

    def run_full_etl(self) -> Tuple[pl.DataFrame, DataQualityReport]:
        """
        Execute the complete ETL pipeline.

        Returns
        -------
        Tuple[pl.DataFrame, DataQualityReport]
            Final dataset and quality report.

        """
        logger.info("Starting full ETL process...")

        try:
            df = self.create_company_profiles()
            df = self.add_derived_metrics(df)
            df = self.standardize_columns(df)
            report = self.generate_data_quality_report(df)

            if self.config.output_path:
                logger.info(f"Saving to: {self.config.output_path}")
                df.write_csv(self.config.output_path)

            self._print_summary(report)
            logger.info("ETL process completed successfully!")

            return df, report

        except Exception as e:
            logger.error(f"ETL process failed: {e}")
            raise ETLException(f"ETL pipeline failed: {e}")

    def _print_summary(self, report: DataQualityReport):
        """
        Print ETL summary.

        Parameters
        ----------
        report : DataQualityReport
            Quality report to summarize.

        """
        print("\n" + "=" * 80)
        print(f"{self.__class__.__name__} COMPLETED")
        print("=" * 80)
        print(f"Total records: {report.total_records:,}")
        print(f"Unique entities: {report.unique_entities:,}")
        print(f"Unique statements: {report.unique_statements:,}")
        print("=" * 80)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def normalize_company_name(name: str) -> str:
    """
    Normalize company name for matching.

    Parameters
    ----------
    name : str
        Company name to normalize.

    Returns
    -------
    str
        Normalized company name.

    """
    if not name:
        return ""

    normalized = name.lower()
    normalized = normalized.strip()

    # Remove common suffixes
    suffixes = [
        "inc",
        "ltd",
        "corp",
        "corporation",
        "company",
        "co",
        "limited",
        "ltee",
        "ltda",
        "pty",
        "plc",
    ]

    for suffix in suffixes:
        normalized = normalized.replace(f" {suffix}", "")
        normalized = normalized.replace(f" {suffix}.", "")

    # Remove punctuation and extra spaces
    import re

    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def log_join_statistics(
    source_df: pl.DataFrame,
    joined_df: pl.DataFrame,
    join_column: str,
    match_column: str,
) -> Dict[str, Any]:
    """
    Calculate and log join statistics.

    Parameters
    ----------
    source_df : pl.DataFrame
        Original source dataframe.
    joined_df : pl.DataFrame
        Joined result dataframe.
    join_column : str
        Column used for joining.
    match_column : str
        Column to check for successful matches.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing join statistics.

    """
    total = len(source_df)
    successful = joined_df.filter(pl.col(match_column).is_not_null()).height
    rate = (successful / total * 100) if total > 0 else 0

    stats = {
        "total_records": total,
        "successful_joins": successful,
        "failed_joins": total - successful,
        "join_rate": round(rate, 1),
        "join_column": join_column,
    }

    logger.info(f"Join Statistics:")
    logger.info(f"  Total records: {total:,}")
    logger.info(f"  Successful joins: {successful:,}")
    logger.info(f"  Join rate: {rate:.1f}%")

    return stats
