"""
Australian modern slavery statements adapter.
"""

import csv
import io
import logging
from typing import List, Dict, Any

from bs4 import BeautifulSoup

from ..base_crawler import BaseCrawler
from ..config import CrawlerConfig
from ..configs.australian import create_config
from ..utils import make_request_with_retry, create_session
from ..utils.html import parse_html
from ..utils.streaming_logger import StreamingLogger

logger = logging.getLogger(__name__)


# Available countries for filtering Australian statements
# Extracted from https://modernslaveryregister.gov.au/statements/
# Format: (country_name, expected_statement_count)
AVAILABLE_COUNTRIES = [
    ("Australia", 11988),
    ("United States of America", 237),
    ("United Kingdom", 184),
    ("New Zealand", 128),
    ("Japan", 74),
    ("Germany", 52),
    ("Singapore", 43),
    ("Canada", 39),
    ("France", 39),
    ("Netherlands", 33),
]

# Extract just the country names for easy iteration
COUNTRY_NAMES = [country[0] for country in AVAILABLE_COUNTRIES]


class AustralianAdapter(BaseCrawler):
    """
    Adapter for Australian modern slavery statements from the Australian Modern Slavery Register.
    
    This adapter handles the CSV-based API endpoint rather than HTML scraping,
    as the Australian system provides direct CSV downloads.
    """

    def __init__(self, config: CrawlerConfig = None, country: str = None):
        """
        Initialize Australian adapter.

        Parameters
        ----------
        config : CrawlerConfig, optional
            Optional custom configuration. If None, uses default Australian config.
        country : str, optional
            Country to filter statements by (e.g., "United States of America").
            If None, downloads all statements.
        """
        if config is None:
            config = create_config(country=country)
        
        super().__init__(config)
        self.country_filter = country

    def extract_statement_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract statement links from Australian system.
        
        Note: Australian system doesn't use HTML parsing for links.
        This method is not used in the Australian workflow but required by base class.

        Parameters
        ----------
        soup : BeautifulSoup
            BeautifulSoup object (not used for Australian system).

        Returns
        -------
        List[str]
            Empty list as Australian system uses direct CSV download.
        """
        logger.debug("Australian system uses direct CSV download, not link extraction")
        return []

    def extract_statement_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract PDF link from an Australian statement page.
        
        This method is used in the second stage of Australian crawling
        to extract direct PDF links from individual statement pages.

        Parameters
        ----------
        soup : BeautifulSoup
            BeautifulSoup object of the statement page.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing PDF URL if found.
        """
        data = {}
        
        # Extract PDF URL from Australian statement page
        pdf_url = self._extract_australian_pdf_url(soup)
        if pdf_url:
            data["pdf_url"] = pdf_url
            
        return data

    def download_csv_data(self, output_filename: str = None) -> List[Dict[str, Any]]:
        """
        Download CSV data directly from Australian Modern Slavery Register.
        
        This is the main method for Australian data extraction, bypassing
        the standard HTML crawling workflow.

        Parameters
        ----------
        output_filename : str, optional
            Optional filename to save raw CSV data.

        Returns
        -------
        List[Dict[str, Any]]
            List of statement records as dictionaries.
        """
        logger.info("Downloading Australian statements data")
        if self.country_filter:
            logger.info(f"Filtering by country: {self.country_filter}")

        # Create session with proper headers
        session = create_session(self.config.custom_headers)
        
        # Add referer header (important for Australian system)
        referer_url = self.config.search_url_template.replace("&csv=", "")
        session.headers.update({"Referer": referer_url})
        
        try:
            # Make the request to download CSV
            response = make_request_with_retry(
                session=session,
                url=self.config.search_url_template,
                timeout=self.config.request_timeout,
                max_retries=3,
                delay=self.config.delay_between_requests
            )
            
            if not response:
                logger.error("Failed to download CSV data")
                return []
            
            # Verify we got CSV content
            content_type = response.headers.get('content-type', '')
            if 'text/csv' not in content_type.lower():
                logger.error(f"Expected CSV, got content-type: {content_type}")
                return []
            
            # Parse CSV content
            csv_content = response.text
            logger.info(f"Downloaded CSV with {len(csv_content)} characters")
            
            # Save raw CSV if filename provided
            if output_filename:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(csv_content)
                logger.info(f"Saved raw CSV to: {output_filename}")
            
            # Parse CSV into list of dictionaries
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            statements = []
            
            for row in csv_reader:
                # Clean and process the row data
                cleaned_row = self._process_csv_row(row)
                statements.append(cleaned_row)
            
            logger.info(f"Processed {len(statements)} statement records")
            return statements
            
        except Exception as e:
            logger.error(f"Error downloading Australian CSV data: {e}")
            return []

    def _process_csv_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """
        Process and clean a single CSV row from Australian data.

        Parameters
        ----------
        row : Dict[str, str]
            Raw CSV row data.

        Returns
        -------
        Dict[str, Any]
            Processed and cleaned row data.
        """
        # Map Australian CSV fields to standardized names
        processed = {
            'idx': row.get('IDX', '').strip(),
            'period_start': row.get('PeriodStart', '').strip(),
            'period_end': row.get('PeriodEnd', '').strip(),
            'statement_type': row.get('Type', '').strip(),
            'headquartered_countries': row.get('HeadquarteredCountries', '').strip(),
            'annual_revenue': row.get('AnnualRevenue', '').strip(),
            'reporting_entities': row.get('ReportingEntities', '').strip(),
            'included_entities': row.get('IncludedEntities', '').strip(),
            'abn': row.get('ABN', '').strip(),
            'acn': row.get('ACN', '').strip(),
            'arbn': row.get('ARBN', '').strip(),
            'statement_url': row.get('Link', '').strip(),
            'industry_sectors': row.get('IndustrySectors', '').strip(),
            'related_statements': row.get('RelatedStatements', '').strip(),
        }
        
        # Add metadata
        processed['jurisdiction'] = 'AU'
        processed['source_system'] = 'Australian Modern Slavery Register'
        processed['country_filter'] = self.country_filter
        
        # Clean empty fields
        processed = {k: v for k, v in processed.items() if v}
        
        return processed

    def _extract_australian_pdf_url(self, soup: BeautifulSoup) -> str:
        """
        Extract PDF URL from Australian statement page.
        
        Australian statement pages embed PDFs in iframes within "The Statement" section.
        The iframe src attribute contains the direct PDF URL.

        Parameters
        ----------
        soup : BeautifulSoup
            BeautifulSoup object of the statement page.

        Returns
        -------
        str or None
            PDF URL if found, None otherwise.
        """
        try:
            # Strategy 1: Look for iframe in "The Statement" section
            # Find the accordion section with "The Statement" 
            statement_sections = soup.find_all('div', {'id': 'the-statement-file'})
            for section in statement_sections:
                iframe = section.find('iframe')
                if iframe and iframe.get('src'):
                    iframe_src = iframe['src']
                    # Convert relative URL to absolute URL
                    if iframe_src.startswith('/'):
                        return f"https://modernslaveryregister.gov.au{iframe_src}"
                    return iframe_src
            
            # Strategy 2: Look for any iframe with PDF-related src
            iframes = soup.find_all('iframe')
            for iframe in iframes:
                src = iframe.get('src', '')
                if src and ('pdf' in src.lower() or '/statements/' in src):
                    # Convert relative URL to absolute URL
                    if src.startswith('/'):
                        return f"https://modernslaveryregister.gov.au{src}"
                    return src
            
            # Strategy 3: Look for download links in statement file actions
            download_links = soup.find_all('a', class_='au-btn')
            for link in download_links:
                if link.get_text(strip=True).lower() == 'download':
                    href = link.get('href')
                    if href:
                        # Convert relative URL to absolute URL
                        if href.startswith('/'):
                            return f"https://modernslaveryregister.gov.au{href}"
                        return href
            
            # Strategy 4: Fallback - look for any PDF links
            pdf_links = soup.find_all("a", href=lambda href: href and ('/pdf/' in href or href.endswith('.pdf')))
            if pdf_links:
                href = pdf_links[0]['href']
                # Convert relative URL to absolute URL
                if href.startswith('/'):
                    return f"https://modernslaveryregister.gov.au{href}"
                return href
                        
        except Exception as e:
            logger.debug(f"Error extracting PDF URL: {e}")
            
        return None

    def crawl_and_save(
        self,
        output_filename: str,
        file_format: str = "csv",
    ) -> List[Dict[str, Any]]:
        """
        Execute two-stage Australian crawl workflow with PDF extraction.
        
        Stage 1: Download CSV data from Australian API
        Stage 2: Crawl individual statement pages to extract PDF links
        
        Includes streaming support similar to Canadian crawler.

        Parameters
        ----------
        output_filename : str
            Output file name for processed data.
        file_format : str, default "csv"
            Output format ("csv" or "json").

        Returns
        -------
        List[Dict[str, Any]]
            List of extracted statement data with PDF URLs.
        """
        import os
        from ..utils import save_to_csv, save_to_json
        
        logger.info(f"Starting two-stage Australian crawl workflow")
        
        # Create output directory
        output_dir = os.path.dirname(output_filename) or "."
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize streaming logger
        base_filename = os.path.splitext(os.path.basename(output_filename))[0]
        streaming_log_file = os.path.join(output_dir, f"{base_filename}_crawl.log")
        streaming_logger = StreamingLogger(streaming_log_file)
        print(f"📝 Streaming results to: {streaming_log_file}")
        
        # Stage 1: Download CSV data
        logger.info("🔄 Stage 1: Downloading CSV data from Australian API")
        raw_csv_filename = output_filename.replace('.csv', '_raw.csv').replace('.json', '_raw.csv')
        csv_data = self.download_csv_data(raw_csv_filename)
        
        if not csv_data:
            logger.error("❌ Stage 1 failed: no CSV data retrieved")
            return []
            
        logger.info(f"✅ Stage 1 completed: {len(csv_data)} statements found in CSV")
        
        # Extract statement URLs for Stage 2
        statement_urls = []
        for row in csv_data:
            statement_url = row.get('statement_url')
            if statement_url:
                statement_urls.append(statement_url)
        
        if not statement_urls:
            logger.warning("⚠️  No statement URLs found in CSV data - returning CSV data only")
            # Save CSV data without PDF URLs
            if file_format.lower() == "json":
                save_to_json(csv_data, output_filename)
            else:
                save_to_csv(csv_data, output_filename)
            return csv_data
        
        logger.info(f"🔄 Stage 2: Crawling {len(statement_urls)} statement pages for PDF links")
        
        # Stage 2: Crawl individual statement pages
        session = create_session(self.config.custom_headers)
        enriched_data = []
        successful_extractions = 0
        
        for i, (csv_row, statement_url) in enumerate(zip(csv_data, statement_urls)):
            try:
                # Make request to statement page
                response = make_request_with_retry(
                    session=session,
                    url=statement_url,
                    timeout=self.config.request_timeout,
                    max_retries=3,
                    delay=self.config.delay_between_requests
                )
                
                if response and response.status_code == 200:
                    # Parse HTML and extract PDF URL
                    soup = parse_html(response.text)
                    pdf_data = self.extract_statement_data(soup)
                    
                    # Merge CSV data with PDF URL
                    enriched_row = csv_row.copy()
                    if pdf_data.get('pdf_url'):
                        enriched_row['pdf_url'] = pdf_data['pdf_url']
                        successful_extractions += 1
                        
                        # Stream the result
                        streaming_logger.log_statement({
                            'title': enriched_row.get('reporting_entities', 'Unknown'),
                            'url': pdf_data['pdf_url']
                        })
                    else:
                        logger.debug(f"No PDF URL found for {statement_url}")
                    
                    enriched_data.append(enriched_row)
                else:
                    logger.warning(f"Failed to fetch {statement_url}")
                    enriched_data.append(csv_row)  # Keep original data
                
                # Progress logging
                if (i + 1) % 10 == 0 or i + 1 == len(statement_urls):
                    progress = (i + 1) / len(statement_urls) * 100
                    logger.info(f"📊 Progress: {i + 1}/{len(statement_urls)} pages processed ({progress:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Error processing {statement_url}: {e}")
                enriched_data.append(csv_row)  # Keep original data
        
        logger.info(f"✅ Stage 2 completed: {successful_extractions}/{len(statement_urls)} PDF URLs extracted ({successful_extractions/len(statement_urls)*100:.1f}% success rate)")
        
        # Save enriched data
        if enriched_data:
            if file_format.lower() == "json":
                save_to_json(enriched_data, output_filename)
            else:
                save_to_csv(enriched_data, output_filename)
            
            logger.info(f"🎉 Australian two-stage crawl completed: {len(enriched_data)} records saved to {output_filename}")
            return enriched_data
        else:
            logger.error("❌ Two-stage crawl failed: no data to save")
            return []

    def crawl_all_countries(self, output_dir: str = "output/australian", file_format: str = "csv") -> Dict[str, List[Dict[str, Any]]]:
        """
        Crawl all available countries sequentially and save separate files.

        Parameters
        ----------
        output_dir : str, default "output/australian"
            Output directory for all country files.
        file_format : str, default "csv"
            Output format ("csv" or "json").

        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Dictionary mapping country names to their statement data.
        """
        import os
        from datetime import datetime
        
        logger.info(f"🌍 Starting multi-country Australian crawl for {len(COUNTRY_NAMES)} countries")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        total_statements = 0
        successful_countries = 0
        failed_countries = 0
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for i, country in enumerate(COUNTRY_NAMES, 1):
            expected_count = dict(AVAILABLE_COUNTRIES)[country]
            
            logger.info(f"📍 [{i}/{len(COUNTRY_NAMES)}] Processing {country} (expected: {expected_count} statements)")
            
            try:
                # Create country-specific adapter
                country_adapter = AustralianAdapter(country=country)
                
                # Generate country-specific filename
                country_code = country.replace(' ', '_').replace('/', '_').lower()
                filename = f"au_{country_code}_{timestamp}.{file_format}"
                output_file = os.path.join(output_dir, filename)
                
                # Crawl this country
                data = country_adapter.crawl_and_save(output_file, file_format)
                
                if data:
                    all_results[country] = data
                    actual_count = len(data)
                    total_statements += actual_count
                    successful_countries += 1
                    
                    # Compare with expected count
                    if actual_count == expected_count:
                        logger.info(f"✅ {country}: {actual_count} statements (matches expected)")
                    else:
                        logger.warning(f"⚠️  {country}: {actual_count} statements (expected {expected_count})")
                else:
                    logger.error(f"❌ {country}: No data retrieved")
                    failed_countries += 1
                    all_results[country] = []
                    
            except Exception as e:
                logger.error(f"❌ {country}: Failed with error: {e}")
                failed_countries += 1
                all_results[country] = []
            
            # Add delay between countries to be respectful
            if i < len(COUNTRY_NAMES):
                logger.info(f"⏳ Waiting {self.config.delay_between_requests}s before next country...")
                import time
                time.sleep(self.config.delay_between_requests)
        
        # Generate summary
        logger.info(f"\n🎉 Multi-country crawl completed!")
        logger.info(f"📊 Summary:")
        logger.info(f"   • Countries processed: {len(COUNTRY_NAMES)}")
        logger.info(f"   • Successful: {successful_countries}")
        logger.info(f"   • Failed: {failed_countries}")
        logger.info(f"   • Total statements: {total_statements:,}")
        logger.info(f"   • Output directory: {output_dir}")
        
        # Create summary file
        summary_file = os.path.join(output_dir, f"au_summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Australian Modern Slavery Statements - Multi-Country Crawl Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 70 + "\n\n")
            
            f.write(f"Overall Statistics:\n")
            f.write(f"  Countries processed: {len(COUNTRY_NAMES)}\n")
            f.write(f"  Successful: {successful_countries}\n")
            f.write(f"  Failed: {failed_countries}\n")
            f.write(f"  Total statements: {total_statements:,}\n\n")
            
            f.write(f"Country Breakdown:\n")
            for country in COUNTRY_NAMES:
                expected_count = dict(AVAILABLE_COUNTRIES)[country]
                actual_count = len(all_results.get(country, []))
                status = "✅" if actual_count > 0 else "❌"
                f.write(f"  {status} {country:<25} {actual_count:>6} / {expected_count:>6}\n")
        
        logger.info(f"📄 Summary saved to: {summary_file}")
        
        return all_results

    @classmethod
    def get_available_countries(cls) -> List[str]:
        """
        Get list of available countries for filtering.

        Returns
        -------
        List[str]
            List of available country names.
        """
        return COUNTRY_NAMES.copy()

    @classmethod
    def get_country_info(cls) -> List[tuple]:
        """
        Get detailed country information including expected statement counts.

        Returns
        -------
        List[tuple]
            List of (country_name, expected_count) tuples.
        """
        return AVAILABLE_COUNTRIES.copy()
