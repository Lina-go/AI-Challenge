"""
Canadian modern slavery statements adapter.
"""

import re
import logging
from typing import List, Dict, Any

from bs4 import BeautifulSoup

from ..base_crawler import BaseCrawler
from ..config import CrawlerConfig, get_config

logger = logging.getLogger(__name__)


# Country indicators for Canadian statements
# Based on company name patterns, legal entity types, and common naming conventions
COUNTRY_INDICATORS = {
    "Canada": [
        "Canada", "Canadian", "Ontario", "Quebec", "Alberta", "British Columbia", "B.C.",
        "Manitoba", "Saskatchewan", "Nova Scotia", "New Brunswick", "Newfoundland",
        "Prince Edward Island", "Northwest Territories", "Nunavut", "Yukon"
    ],
    "United States": [
        "USA", "U.S.", "United States", "American", "Delaware", "Nevada", "California",
        "New York", "Texas", "Florida", "Illinois", "Corp.", "LLC", "Inc."
    ],
    "Netherlands": [
        "B.V.", "N.V.", "Netherlands", "Dutch", "Holland", "Amsterdam"
    ],
    "United Kingdom": [
        "UK", "U.K.", "British", "England", "Scotland", "Wales", "London", "plc", "Limited"
    ],
    "Germany": [
        "GmbH", "AG", "German", "Germany", "Deutschland", "Berlin", "Munich"
    ],
    "France": [
        "S.A.", "S.A.S.", "SARL", "French", "France", "Paris", "Lyon"
    ],
    "Japan": [
        "K.K.", "Japanese", "Japan", "Tokyo", "Osaka", "Co., Ltd."
    ],
    "Australia": [
        "Pty Ltd", "Australian", "Australia", "Sydney", "Melbourne"
    ],
    "Switzerland": [
        "AG", "SA", "Swiss", "Switzerland", "Zurich", "Geneva"
    ],
    "Sweden": [
        "AB", "Swedish", "Sweden", "Stockholm"
    ],
}


class CanadianAdapter(BaseCrawler):
    """
    Adapter for Canadian modern slavery statements from Public Safety Canada.

    This adapter handles the specific HTML structure and data extraction
    patterns used by the Canadian government's statement repository.
    """

    def __init__(self, config: CrawlerConfig = None):
        """
        Initialize Canadian adapter.

        Parameters
        ----------
        config : CrawlerConfig, optional
            Optional custom configuration. If None, uses default Canadian config.
        """
        if config is None:
            config = get_config("canadian")

        super().__init__(config)

    def extract_statement_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract statement page links from Canadian search results.

        The Canadian site uses table rows with anchor tags containing
        relative URLs to individual statement pages.

        Parameters
        ----------
        soup : BeautifulSoup
            BeautifulSoup object of the search page.

        Returns
        -------
        List[str]
            List of relative URLs to statement pages.
        """
        links = []

        try:
            # Find all table rows and extract anchor tags
            rows = soup.find_all("tr")
            for row in rows:
                link_tag = row.find("a", href=True)
                if link_tag and link_tag.get("href"):
                    href = link_tag["href"]
                    # Only include statement detail pages (dtls-en.aspx)
                    if "dtls-en.aspx" in href and "d=PS&i=" in href:
                        links.append(href)

            logger.debug(f"Extracted {len(links)} links from search page")

        except Exception as e:
            logger.error(f"Failed to extract links: {e}")

        return links

    def extract_statement_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract data from a Canadian statement page.

        Canadian pages typically contain:
        - Title in h2 tags (often with year in parentheses)
        - PDF link with text "Online access"
        - Additional metadata in various elements

        Parameters
        ----------
        soup : BeautifulSoup
            BeautifulSoup object of the statement page.

        Returns
        -------
        Dict[str, Any]
            Dictionary of extracted data.
        """
        data = {}

        try:
            # Extract title
            title = self._extract_title(soup)
            if title:
                data["title"] = title

            # Extract PDF URL
            pdf_url = self._extract_pdf_url(soup)
            if pdf_url:
                data["pdf_url"] = pdf_url

            # Extract additional metadata
            year = self._extract_year(soup, title)
            if year:
                data["year"] = year

            organization = self._extract_organization(soup, title)
            if organization:
                data["organization"] = organization

            language = self._detect_language(soup)
            if language:
                data["language"] = language

            # Extract file type
            if pdf_url:
                data["file_type"] = self._extract_file_type(pdf_url)

            # Extract enhanced metadata
            legal_info = self._extract_legal_framework(soup)
            data.update(legal_info)
            
            industry_info = self._extract_industry_classification(soup)
            data.update(industry_info)
            
            company_info = self._extract_company_details(soup)
            data.update(company_info)
            
            document_type = self._extract_document_type(soup)
            if document_type:
                data["document_type"] = document_type

        except Exception as e:
            logger.error(f"Failed to extract statement data: {e}")

        return data

    def _detect_company_country(self, statement_data: Dict[str, Any]) -> str:
        """
        Detect the likely country of origin for a company based on various indicators.

        Parameters
        ----------
        statement_data : Dict[str, Any]
            Statement data dictionary.

        Returns
        -------
        str
            Detected country name or "Canada" as default.
        """
        # Check if we already have extracted country info
        if 'company_country' in statement_data:
            return statement_data['company_country']
        
        # Text to analyze for country indicators
        analysis_text = []
        
        # Add various fields for analysis
        for field in ['title', 'organization', 'statement_name', 'parent_company']:
            if field in statement_data and statement_data[field]:
                analysis_text.append(str(statement_data[field]))
        
        combined_text = " ".join(analysis_text)
        
        # Score each country based on indicators found
        country_scores = {}
        
        for country, indicators in COUNTRY_INDICATORS.items():
            score = 0
            for indicator in indicators:
                # Case-insensitive search with word boundaries where appropriate
                if indicator in combined_text:
                    # Give higher scores to more specific indicators
                    if len(indicator) > 3:
                        score += 2
                    else:
                        score += 1
            
            if score > 0:
                country_scores[country] = score
        
        # Return the country with the highest score, default to Canada
        if country_scores:
            return max(country_scores.items(), key=lambda x: x[1])[0]
        
        return "Canada"  # Default for Canadian jurisdiction

    def filter_by_country(self, data: List[Dict[str, Any]], country_filter: str) -> List[Dict[str, Any]]:
        """
        Filter Canadian statements by detected company country.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            List of statement data dictionaries.
        country_filter : str
            Country to filter by.

        Returns
        -------
        List[Dict[str, Any]]
            Filtered list of statements.
        """
        if not country_filter:
            return data
        
        filtered_data = []
        
        for statement in data:
            detected_country = self._detect_company_country(statement)
            
            # Add detected country to the statement data
            statement['detected_country'] = detected_country
            
            # Filter based on country
            if detected_country.lower() == country_filter.lower():
                filtered_data.append(statement)
        
        logger.info(f"Filtered {len(data)} statements to {len(filtered_data)} for country: {country_filter}")
        return filtered_data

    def crawl_and_save_with_country_filter(
        self,
        output_filename: str,
        file_format: str = "csv",
        country_filter: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute Canadian crawl with optional country filtering.

        Parameters
        ----------
        output_filename : str
            Output file name.
        file_format : str, default "csv"
            Output format ("csv" or "json").
        country_filter : str, optional
            Country to filter statements by.

        Returns
        -------
        List[Dict[str, Any]]
            List of extracted and optionally filtered statement data.
        """
        # First, do the regular crawl
        data = self.crawl_and_save(output_filename, file_format)
        
        # If country filter is specified, apply post-processing filter
        if country_filter and data:
            filtered_data = self.filter_by_country(data, country_filter)
            
            # Save filtered data to a new file
            if filtered_data:
                import os
                base_name, ext = os.path.splitext(output_filename)
                country_code = country_filter.replace(' ', '_').lower()
                filtered_filename = f"{base_name}_{country_code}{ext}"
                
                from ..utils import save_to_csv, save_to_json
                
                if file_format.lower() == "json":
                    save_to_json(filtered_data, filtered_filename)
                else:
                    save_to_csv(filtered_data, filtered_filename)
                
                logger.info(f"Saved {len(filtered_data)} filtered statements to: {filtered_filename}")
                return filtered_data
            else:
                logger.warning(f"No statements found for country: {country_filter}")
                return []
        
        return data

    @classmethod
    def get_available_countries(cls) -> List[str]:
        """
        Get list of countries that can be detected in Canadian statements.

        Returns
        -------
        List[str]
            List of detectable country names.
        """
        return list(COUNTRY_INDICATORS.keys())

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the statement title from the page."""
        # Strategy 1: Look for h2 tags with year patterns
        h2_tags = soup.find_all("h2")
        for tag in h2_tags:
            title_text = tag.get_text(strip=True)
            # Check if it contains a year pattern
            if re.search(r"\b(20\d{2})\b", title_text):
                return title_text

        # Strategy 2: Look for h2 tags that aren't navigation
        for tag in h2_tags:
            title_text = tag.get_text(strip=True)
            if not any(
                skip in title_text.lower()
                for skip in ["language", "search", "menu", "navigation", "breadcrumb"]
            ):
                return title_text

        # Strategy 3: Use page title as fallback
        title_tag = soup.find("title")
        if title_tag:
            page_title = title_tag.get_text(strip=True)
            # Clean up common prefixes/suffixes
            page_title = re.sub(r"^.*?\s*-\s*", "", page_title)  # Remove site prefix
            page_title = re.sub(r"\s*-\s*.*?$", "", page_title)  # Remove site suffix
            if page_title and len(page_title) > 10:  # Ensure it's substantial
                return page_title

        return None

    def _extract_pdf_url(self, soup: BeautifulSoup) -> str:
        """Extract the PDF URL from the page."""
        # Strategy 1: Look for "Online access" link (English)
        online_access_link = soup.find("a", string="Online access")
        if online_access_link and online_access_link.get("href"):
            return online_access_link["href"]

        # Strategy 2: Look for "Accès en ligne" link (French)
        french_access_link = soup.find("a", string="Accès en ligne")
        if french_access_link and french_access_link.get("href"):
            return french_access_link["href"]

        # Strategy 3: Look for any PDF link
        pdf_links = soup.find_all("a", href=re.compile(r"\.pdf$", re.IGNORECASE))
        if pdf_links:
            return pdf_links[0]["href"]

        # Strategy 4: Look for links containing "pdf" in the URL
        pdf_pattern_links = soup.find_all("a", href=re.compile(r"pdf", re.IGNORECASE))
        if pdf_pattern_links:
            return pdf_pattern_links[0]["href"]

        return None

    def _extract_year(self, soup: BeautifulSoup, title: str = None) -> int:
        """Extract the reporting year."""
        # Try to extract from title first
        if title:
            year_match = re.search(r"\b(20\d{2})\b", title)
            if year_match:
                return int(year_match.group(1))

        # Look for year in other page elements
        year_patterns = [
            r"reporting year[:\s]+(\d{4})",
            r"year[:\s]+(\d{4})",
            r"\b(20\d{2})\b",
        ]

        page_text = soup.get_text()
        for pattern in year_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                # Return the most recent year found
                years = [int(year) for year in matches if 2020 <= int(year) <= 2030]
                if years:
                    return max(years)

        return None

    def _extract_organization(self, soup: BeautifulSoup, title: str = None) -> str:
        """Extract the organization name."""
        if not title:
            return None

        # Remove year and common suffixes to get organization name
        org_name = re.sub(r"\s*\(20\d{2}\)\s*", "", title)
        org_name = re.sub(
            r"\s*-?\s*Modern Slavery.*", "", org_name, flags=re.IGNORECASE
        )
        org_name = re.sub(r"\s*-?\s*Statement.*", "", org_name, flags=re.IGNORECASE)
        org_name = org_name.strip()

        # Return only if it's different from the original title and substantial
        if org_name and org_name != title and len(org_name) > 5:
            return org_name

        return None

    def _detect_language(self, soup: BeautifulSoup) -> str:
        """Detect the language of the page."""
        # Check HTML lang attribute
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            lang = html_tag["lang"].lower()
            if "fr" in lang:
                return "fr"
            elif "en" in lang:
                return "en"

        french_indicators = [
            soup.find_all("a", string="Accès en ligne"),
            soup.find_all(text=re.compile(r"\bfrançais\b", re.IGNORECASE)),
        ]

        if any(indicators for indicators in french_indicators):
            return "fr"

        english_indicators = [
            soup.find_all("a", string="Online access"),
            soup.find_all(text=re.compile(r"\benglish\b", re.IGNORECASE)),
        ]

        if any(indicators for indicators in english_indicators):
            return "en"

        return "en"

    def _extract_file_type(self, url: str) -> str:
        """Extract file type from URL."""
        if not url:
            return None

        # Extract extension from URL
        extension_match = re.search(r"\.([a-zA-Z0-9]+)(?:\?|$)", url.lower())
        if extension_match:
            return extension_match.group(1)

        # Default assumption for Canadian site
        return "pdf"
    
    def _extract_legal_framework(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract legal framework and reporting law information.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary with legal framework data
        """
        legal_info = {}
        
        try:
            # Find subject section
            subject_section = soup.find('h3', string='Subject')
            if subject_section and subject_section.find_next_sibling('ul'):
                subjects = subject_section.find_next_sibling('ul').find_all('a')
                
                for subject in subjects:
                    text = subject.get_text().strip()
                    
                    # Identify reporting law
                    if 'Fighting Against Forced Labour' in text and ':' not in text:
                        legal_info['reporting_law'] = text
                    
                    # Identify legal framework code
                    elif text == 'SCALCA':
                        legal_info['legal_framework'] = 'Supply Chains Act (SCALCA)'
                    
                    # Identify if it's mandatory reporting
                    elif 'Annual Reports:' in text:
                        legal_info['statement_type'] = 'mandatory'
        except Exception as e:
            logger.debug(f"Could not extract legal framework: {e}")
        
        return legal_info
    
    def _extract_industry_classification(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract industry sector and subsector information.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary with industry classification data
        """
        industry_info = {}
        
        try:
            # Find subject section for industry classifications
            subject_section = soup.find('h3', string='Subject')
            if subject_section and subject_section.find_next_sibling('ul'):
                subjects = subject_section.find_next_sibling('ul').find_all('a')
                
                # Known industry sectors (could be expanded)
                industry_sectors = [
                    'Manufacturing', 'Construction', 'Retail trade', 'Wholesale trade',
                    'Transportation and warehousing', 'Finance and insurance',
                    'Professional, scientific and technical services', 'Utilities',
                    'Information and cultural industries', 'Accommodation and food services',
                    'Health care and social assistance', 'Educational services',
                    'Arts, entertainment and recreation', 'Other services',
                    'Public administration', 'Mining, quarrying, and oil and gas extraction',
                    'Agriculture, forestry, fishing and hunting'
                ]
                
                found_sectors = []
                found_subsectors = []
                
                for subject in subjects:
                    text = subject.get_text().strip()
                    
                    if text in industry_sectors:
                        found_sectors.append(text)
                    elif any(sector in text for sector in industry_sectors):
                        # This might be a subsector
                        found_subsectors.append(text)
                
                if found_sectors:
                    industry_info['industry_sector'] = found_sectors[0]  # Primary sector
                    if len(found_sectors) > 1:
                        industry_info['industry_sectors_additional'] = found_sectors[1:]
                
                if found_subsectors:
                    industry_info['industry_subsector'] = found_subsectors[0]
                    if len(found_subsectors) > 1:
                        industry_info['industry_subsectors_additional'] = found_subsectors[1:]
        
        except Exception as e:
            logger.debug(f"Could not extract industry classification: {e}")
        
        return industry_info
    
    def _extract_company_details(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract detailed company information including subsidiaries and location.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary with company details
        """
        company_info = {}
        
        try:
            # Extract authors (parent and subsidiary companies)
            authors_section = soup.find('h3', string='Authors')
            if authors_section and authors_section.find_next_sibling('ul'):
                authors = authors_section.find_next_sibling('ul').find_all('a')
                
                author_companies = []
                for author in authors:
                    company_name = author.get_text().strip()
                    # Remove ", author." suffix
                    company_name = re.sub(r',\s*author\.$', '', company_name)
                    author_companies.append(company_name)
                
                if author_companies:
                    company_info['parent_company'] = author_companies[0]
                    if len(author_companies) > 1:
                        company_info['subsidiary_companies'] = author_companies[1:]
            
            # Extract company country from publishers
            publishers_section = soup.find('h3', string='Publishers')
            if publishers_section and publishers_section.find_next_sibling('ul'):
                publishers = publishers_section.find_next_sibling('ul').find_all('li')
                
                for publisher in publishers:
                    text = publisher.get_text().strip()
                    # Look for pattern "Country : Company"
                    if ':' in text:
                        country = text.split(':')[0].strip()
                        company_info['company_country'] = country
                        break
        
        except Exception as e:
            logger.debug(f"Could not extract company details: {e}")
        
        return company_info
    
    def _extract_document_type(self, soup: BeautifulSoup) -> str:
        """
        Extract document type from resource section.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Document type string
        """
        try:
            resource_section = soup.find('h3', string='Resource')
            if resource_section and resource_section.find_next_sibling('p'):
                return resource_section.find_next_sibling('p').get_text().strip()
        except Exception as e:
            logger.debug(f"Could not extract document type: {e}")
        
        return None


def create_canadian_crawler(config: CrawlerConfig = None) -> CanadianAdapter:
    """
    Factory function to create a Canadian crawler.

    Args:
        config: Optional custom configuration. If None, uses default Canadian config.

    Returns:
        CanadianAdapter instance
    """
    return CanadianAdapter(config)
