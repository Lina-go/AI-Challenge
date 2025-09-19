"""
Column Standardization Module
==============================

Provides standardization mappings and utilities for harmonizing column names
across different country datasets.

"""

import polars as pl
from typing import Dict, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StandardColumns(Enum):
    """Standardized column names across all datasets."""
    
    # === IDENTIFIERS ===
    COMPANY_ID = 'company_id'
    COMPANY_NUMBER = 'company_number'
    ABN = 'abn'  # Australian Business Number
    ACN = 'acn'  # Australian Company Number
    BUSINESS_ID = 'business_id'
    STATEMENT_ID = 'statement_id'
    
    # === COMPANY INFORMATION ===
    COMPANY_NAME = 'company_name'
    ORGANIZATION_NAME = 'organization_name'
    COMPANY_NAME_ALT = 'company_name_alt'
    COMPANY_NAME_NORMALIZED = 'company_name_normalized'
    PARENT_COMPANY = 'parent_company'
    SUBSIDIARY_COMPANIES = 'subsidiary_companies'
    
    # === DATES ===
    STATEMENT_YEAR = 'statement_year'
    STATEMENT_START_DATE = 'statement_start_date'
    STATEMENT_END_DATE = 'statement_end_date'
    DATE_APPROVED = 'date_approved'
    DATE_UPDATED = 'date_updated'
    INCORPORATION_DATE = 'incorporation_date'
    DISSOLUTION_DATE = 'dissolution_date'
    
    # === STATUS ===
    COMPANY_STATUS = 'company_status'
    IS_ACTIVE = 'is_active'
    IS_DISSOLVED = 'is_dissolved'
    
    # === STATEMENT URLS ===
    STATEMENT_URL = 'statement_url'
    PDF_URL = 'pdf_url'
    SOURCE_URL = 'source_url'
    
    # === LOCATION ===
    JURISDICTION = 'jurisdiction'
    COUNTRY = 'country'
    HEADQUARTER_COUNTRIES = 'headquarter_countries'
    PROVINCE_STATE = 'province_state'
    CITY = 'city'
    FULL_ADDRESS = 'full_address'
    
    # === INDUSTRY ===
    INDUSTRY_SECTORS = 'industry_sectors'
    BUSINESS_SECTOR = 'business_sector'
    BUSINESS_SUBSECTOR = 'business_subsector'
    NAICS_CODE = 'naics_code'
    SIC_CODE = 'sic_code'
    
    # === COMPLIANCE ===
    STATEMENT_TYPE = 'statement_type'
    HAS_ORG_STRUCTURE = 'has_org_structure'
    HAS_POLICIES = 'has_policies'
    HAS_RISK_ASSESSMENT = 'has_risk_assessment'
    HAS_DUE_DILIGENCE = 'has_due_diligence'
    HAS_TRAINING = 'has_training'
    HAS_GOALS = 'has_goals'
    COMPLIANCE_SCORE = 'compliance_score'
    
    # === FINANCIAL ===
    ANNUAL_REVENUE = 'annual_revenue'
    TURNOVER = 'turnover'
    EMPLOYEE_COUNT = 'employee_count'
    COMPANY_SIZE = 'company_size'
    
    # === METADATA ===
    DATA_SOURCE = 'data_source'
    JOIN_METHOD = 'join_method'
    DATA_COMPLETENESS_SCORE = 'data_completeness_score'
    LANGUAGE = 'language'


class ColumnStandardizer:
    """
    Standardizes column names across different country datasets.
    
    This class provides mappings and methods to convert country-specific
    column names to a standardized format.
    """
    
    # Australia column mappings
    AUSTRALIA_MAPPING = {
        # Identifiers
        'ABN': 'abn',
        'ACN': 'acn',
        'idx': 'statement_id',
        'statement_group_id': 'statement_group_id',
        
        # Dates
        'period_start': 'statement_start_date',
        'period_end': 'statement_end_date',
        'statement_year': 'statement_year',
        'Date of Registration': 'incorporation_date',
        'Date of Deregistration': 'dissolution_date',
        'Current Name Start Date': 'current_name_start_date',
        
        # Company Info
        'reporting_entities': 'organization_name',
        'Company Name': 'company_name',
        'Current Name': 'current_company_name',
        'entity_name_normalized': 'organization_name_normalized',
        'company_name_normalized': 'company_name_normalized',
        'included_entities': 'subsidiary_companies',
        
        # Status
        'Status': 'company_status',
        'is_active': 'is_active',
        'is_deregistered': 'is_dissolved',
        'Current Name Indicator': 'is_current_name',
        
        # URLs
        'statement_url': 'statement_url',
        'pdf_url': 'pdf_url',
        
        # Location
        'jurisdiction': 'jurisdiction',
        'headquartered_countries': 'headquarter_countries',
        'country_filter': 'country_filter',
        'Previous State of Registration': 'previous_state_registration',
        'State Registration number': 'state_registration_number',
        
        # Industry
        'industry_sectors': 'industry_sectors',
        'Type': 'company_type',
        'Class': 'company_class',
        'Sub Class': 'company_sub_class',
        
        # Financial
        'annual_revenue': 'annual_revenue',
        
        # Metadata
        'statement_type': 'statement_type',
        'source_system': 'data_source',
        'join_method': 'join_method',
        'data_completeness_score': 'data_completeness_score',
        'is_primary_entity': 'is_primary_entity',
        'group_entity_count': 'group_entity_count',
        'entity_index_in_group': 'entity_index_in_group',
        'Modified since last report': 'modified_since_last_report',
        
        # Derived metrics
        'company_age_at_statement': 'company_age_at_statement',
        'statement_period_days': 'statement_period_days',
        'num_headquarter_countries': 'num_headquarter_countries',
        'match_quality_score': 'match_quality_score',
        'entity_role': 'entity_role',
        
        # Flags
        'has_company_data': 'has_company_data',
        'has_asic_match': 'has_registry_match',
        'matched_by_abn': 'matched_by_abn',
        'is_multi_entity_statement': 'is_multi_entity_statement',
        'has_revenue_data': 'has_revenue_data',
        'has_statement_url': 'has_statement_url',
        'has_pdf_url': 'has_pdf_url',
        'has_industry_data': 'has_industry_data',
    }
    
    # Canada column mappings
    CANADA_MAPPING = {
        # Identifiers
        'idx': 'registry_id',
        'business_id_no': 'business_id',
        'licence_number': 'licence_number',
        
        # Dates
        'year': 'statement_year',
        
        # Company Info
        'organization': 'organization_name',
        'business_name': 'company_name',
        'alt_business_name': 'company_name_alt',
        'parent_company': 'parent_company',
        'subsidiary_companies': 'subsidiary_companies',
        'parent_company_match_key': 'parent_company_normalized',
        'title': 'statement_title',
        'statement_name': 'statement_name',
        
        # Status
        'status': 'company_status',
        'IsActiveBusiness': 'is_active',
        
        # URLs
        'statement_url': 'statement_url',
        'pdf_url': 'pdf_url',
        'source_url': 'source_url',
        
        # Location
        'jurisdiction': 'jurisdiction',
        'company_country': 'country',
        'prov_terr': 'province_state',
        'city': 'city',
        'full_address': 'full_address',
        
        # Industry
        'industry_sector': 'industry_sectors',
        'industry_sectors_additional': 'industry_sectors_additional',
        'industry_subsector': 'industry_subsector',
        'industry_subsectors_additional': 'industry_subsectors_additional',
        'business_sector': 'business_sector',
        'business_subsector': 'business_subsector',
        'business_description': 'business_description',
        'derived_NAICS': 'naics_code',
        'source_NAICS_primary': 'naics_primary',
        'source_NAICS_secondary': 'naics_secondary',
        'NAICS_descr': 'naics_description',
        'NAICS_descr2': 'naics_description_secondary',
        
        # Financial
        'total_no_employees': 'employee_count',
        
        # Metadata
        'statement_type': 'statement_type',
        'language': 'language',
        'file_type': 'file_type',
        'document_type': 'document_type',
        'legal_framework': 'legal_framework',
        'reporting_law': 'reporting_law',
        'licence_type': 'licence_type',
        'provider': 'data_provider',
        'DataCompletenessScore': 'data_completeness_score',
        'RecencyScore': 'recency_score',
        
        # Flags
        'IsEnglish': 'is_english',
        'IsFrench': 'is_french',
        'IsBilingual': 'is_bilingual',
        'HasPDFUrl': 'has_pdf_url',
        'HasStatementUrl': 'has_statement_url',
        'HasParentCompany': 'has_parent_company',
        'HasSubsidiaries': 'has_subsidiaries',
        'MatchedToBizRegistry': 'has_registry_match',
        'HasEmployeeCount': 'has_employee_count',
        'InMajorProvince': 'in_major_province',
        'HasNAICSCode': 'has_naics_code',
    }
    
    # UK column mappings
    UK_MAPPING = {
        # Identifiers
        'CompanyNumber': 'company_number',
        'URI': 'company_uri',
        
        # Dates
        'StatementYear': 'statement_year',
        'StatementStartDate': 'statement_start_date',
        'StatementEndDate': 'statement_end_date',
        'DateApproved': 'date_approved',
        'LastUpdated': 'date_updated',
        'IncorporationDate': 'incorporation_date',
        'DissolutionDate': 'dissolution_date',
        'Accounts.NextDueDate': 'accounts_next_due_date',
        'Accounts.LastMadeUpDate': 'accounts_last_made_date',
        
        # Company Info
        'OrganisationName': 'organization_name',
        'CompanyName': 'company_name',
        'ParentName': 'parent_company',
        'PreviousName_1.CompanyName': 'previous_name_1',
        'PreviousName_2.CompanyName': 'previous_name_2',
        'ApprovingPerson': 'approving_person',
        
        # Status
        'CompanyStatus': 'company_status',
        'IsActive': 'is_active',
        'IsDissolved': 'is_dissolved',
        
        # URLs
        'StatementURL': 'statement_url',
        'PDFURL': 'pdf_url',
        
        # Location
        'Address': 'address',
        'RegAddress.AddressLine1': 'address_line_1',
        'RegAddress.AddressLine2': 'address_line_2',
        'RegAddress.PostTown': 'post_town',
        'RegAddress.County': 'county',
        'RegAddress.Country': 'country',
        'RegAddress.PostCode': 'post_code',
        'FullRegisteredAddress': 'full_address',
        'CountryOfOrigin': 'country_of_origin',
        
        # Industry
        'OrganisationSectors': 'industry_sectors',
        'SectorType': 'business_sector',
        'SICCode.SicText_1': 'sic_code_1',
        'SICCode.SicText_2': 'sic_code_2',
        'SICCode.SicText_3': 'sic_code_3',
        'SICCode.SicText_4': 'sic_code_4',
        
        # Financial
        'Turnover': 'turnover',
        'CompanySize': 'company_size',
        'Mortgages.NumMortCharges': 'mortgage_charges',
        'Mortgages.NumMortOutstanding': 'mortgages_outstanding',
        'Accounts.AccountCategory': 'account_category',
        
        # Compliance
        'StatementIncludesOrgStructure': 'statement_includes_org_structure',
        'StatementIncludesPolicies': 'statement_includes_policies',
        'StatementIncludesRisksAssessment': 'statement_includes_risks',
        'StatementIncludesDueDiligence': 'statement_includes_due_diligence',
        'StatementIncludesTraining': 'statement_includes_training',
        'StatementIncludesGoals': 'statement_includes_goals',
        'HasOrgStructure': 'has_org_structure',
        'HasPolicies': 'has_policies',
        'HasRiskAssessment': 'has_risk_assessment',
        'HasDueDiligence': 'has_due_diligence',
        'HasTraining': 'has_training',
        'HasGoals': 'has_goals',
        'ComplianceScore': 'compliance_score',
        'HighCompliance': 'is_high_compliance',
        'LowCompliance': 'is_low_compliance',
        
        # Risk Assessment
        'Risk1': 'risk_1',
        'Risk1Area': 'risk_1_area',
        'Risk1Location': 'risk_1_location',
        'Risk1Mitigation': 'risk_1_mitigation',
        'Risk2': 'risk_2',
        'Risk2Area': 'risk_2_area',
        'Risk2Location': 'risk_2_location',
        'Risk2Mitigation': 'risk_2_mitigation',
        'Risk3': 'risk_3',
        'Risk3Area': 'risk_3_area',
        'Risk3Location': 'risk_3_location',
        'Risk3Mitigation': 'risk_3_mitigation',
        'HasRisk1': 'has_risk_1',
        'HasRisk2': 'has_risk_2',
        'HasRisk3': 'has_risk_3',
        'RiskAssessmentCompleteness': 'risk_assessment_completeness',
        
        # Other
        'CompanyCategory': 'company_category',
        'GroupSubmission': 'is_group_submission',
        'YearsProducingStatements': 'years_producing_statements',
        'DemonstrateProgress': 'demonstrates_progress',
        'HasProgressDemonstration': 'has_progress_demonstration',
        'HasMortgageCharges': 'has_mortgage_charges',
        'CompanyAgeAtStatement': 'company_age_at_statement',
    }
    
    @classmethod
    def get_mapping(cls, country: str) -> Dict[str, str]:
        """
        Get column mapping for a specific country.
        
        Parameters
        ----------
        country : str
            Country code ('australia', 'canada', 'uk').
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping original to standardized column names.
        
        """
        mappings = {
            'australia': cls.AUSTRALIA_MAPPING,
            'canada': cls.CANADA_MAPPING,
            'uk': cls.UK_MAPPING
        }
        
        if country not in mappings:
            raise ValueError(f"Unknown country: {country}")
        
        return mappings[country]
    
    @classmethod
    def standardize_dataframe(
        cls, 
        df: pl.DataFrame, 
        country: str,
        add_country_column: bool = True
    ) -> pl.DataFrame:
        """
        Standardize column names in a dataframe.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe with country-specific column names.
        country : str
            Country code ('australia', 'canada', 'uk').
        add_country_column : bool
            Whether to add a 'data_country' column.
        
        Returns
        -------
        pl.DataFrame
            Dataframe with standardized column names.
        
        Examples
        --------
        >>> df = pl.DataFrame({'ABN': [1, 2], 'Company Name': ['A', 'B']})
        >>> df_std = ColumnStandardizer.standardize_dataframe(df, 'australia')
        >>> print(df_std.columns)
        ['abn', 'company_name', 'data_country']
        
        """
        mapping = cls.get_mapping(country)
        
        # Create rename mapping only for columns that exist
        rename_map = {}
        for old_name, new_name in mapping.items():
            if old_name in df.columns:
                rename_map[old_name] = new_name
        
        # Log the renaming
        if rename_map:
            logger.info(f"Renaming {len(rename_map)} columns for {country}")
            logger.debug(f"Rename mapping: {rename_map}")
        
        # Apply renaming
        df = df.rename(rename_map)
        
        # Add country identifier
        if add_country_column:
            df = df.with_columns([
                pl.lit(country).alias('data_country')
            ])
        
        # Convert remaining columns to snake_case
        df = cls.convert_to_snake_case(df)
        
        return df
    
    @staticmethod
    def convert_to_snake_case(df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert all column names to snake_case.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe.
        
        Returns
        -------
        pl.DataFrame
            Dataframe with snake_case column names.
        
        """
        import re
        
        def to_snake_case(name: str) -> str:
            # Replace dots and spaces with underscores
            name = name.replace('.', '_').replace(' ', '_')
            # Insert underscore before capitals (for camelCase)
            name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
            # Convert to lowercase and remove double underscores
            name = name.lower()
            name = re.sub('_+', '_', name)
            # Remove leading/trailing underscores
            return name.strip('_')
        
        rename_map = {}
        for col in df.columns:
            snake_case_name = to_snake_case(col)
            if col != snake_case_name:
                rename_map[col] = snake_case_name
        
        if rename_map:
            df = df.rename(rename_map)
        
        return df
    
    @staticmethod
    def get_common_columns(*dataframes: pl.DataFrame) -> List[str]:
        """
        Get columns that are common across multiple dataframes.
        
        Parameters
        ----------
        *dataframes : pl.DataFrame
            Variable number of dataframes to compare.
        
        Returns
        -------
        List[str]
            List of column names present in all dataframes.
        
        """
        if not dataframes:
            return []
        
        common = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common &= set(df.columns)
        
        return sorted(list(common))
    
    @staticmethod
    def align_dataframes(*dataframes: pl.DataFrame) -> List[pl.DataFrame]:
        """
        Align multiple dataframes to have the same columns.
        
        Parameters
        ----------
        *dataframes : pl.DataFrame
            Variable number of dataframes to align.
        
        Returns
        -------
        List[pl.DataFrame]
            List of aligned dataframes with same columns.
        
        """
        if not dataframes:
            return []
        
        # Get all unique columns
        all_columns = set()
        for df in dataframes:
            all_columns.update(df.columns)
        
        all_columns = sorted(list(all_columns))
        
        # Align each dataframe
        aligned = []
        for df in dataframes:
            # Add missing columns with null values
            missing_cols = set(all_columns) - set(df.columns)
            if missing_cols:
                null_cols = [pl.lit(None).alias(col) for col in missing_cols]
                df = df.with_columns(null_cols)
            
            # Select columns in consistent order
            df = df.select(all_columns)
            aligned.append(df)
        
        return aligned


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

def apply_standardization_to_etl():
    """
    Example of where to apply standardization in the ETL pipeline.
    
    The standardization should be applied AFTER all derived metrics
    are added but BEFORE saving the final output.
    
    """
    # In australia_etl.py, modify the run_full_etl method:
    """
    def run_full_etl(self) -> Tuple[pl.DataFrame, DataQualityReport]:
        logger.info("Starting full ETL process...")
        
        # Create company profiles
        df = self.create_company_profiles()
        
        # Add derived metrics
        df = self.add_derived_metrics(df)
        
        # === ADD STANDARDIZATION HERE ===
        from column_standardization import ColumnStandardizer
        df = ColumnStandardizer.standardize_dataframe(df, 'australia')
        
        # Generate quality report (may need to update field names in report)
        quality_report = self.generate_data_quality_report(df)
        
        # Save if output file specified
        if self.config.output_path:
            logger.info(f"Saving to: {self.config.output_path}")
            df.write_csv(self.config.output_path)
        
        return df, quality_report
    """
    
    # Same pattern for canada_etl.py and uk_etl.py
    pass


def combine_standardized_datasets():
    """
    Example of combining standardized datasets from multiple countries.
    
    """
    from column_standardization import ColumnStandardizer
    
    # Load datasets (after ETL processing)
    au_df = pl.read_csv("output/australia_enriched.csv")
    ca_df = pl.read_csv("output/canada_enriched.csv")
    uk_df = pl.read_csv("output/uk_enriched.csv")
    
    # They should already be standardized if apply_standardization_to_etl was used
    # But we can verify/ensure standardization:
    au_df = ColumnStandardizer.standardize_dataframe(au_df, 'australia')
    ca_df = ColumnStandardizer.standardize_dataframe(ca_df, 'canada')
    uk_df = ColumnStandardizer.standardize_dataframe(uk_df, 'uk')
    
    # Get common columns
    common_cols = ColumnStandardizer.get_common_columns(au_df, ca_df, uk_df)
    print(f"Common columns across all datasets: {common_cols}")
    
    # Align dataframes (add missing columns as nulls)
    aligned_dfs = ColumnStandardizer.align_dataframes(au_df, ca_df, uk_df)
    
    # Now they can be combined
    combined_df = pl.concat(aligned_dfs, how="vertical")
    
    # Save combined dataset
    combined_df.write_csv("output/all_countries_combined.csv")
    
    return combined_df