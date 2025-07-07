import pandas as pd
import requests
import time
import logging
from urllib.parse import urlparse
from typing import Optional, Dict
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebsiteValidator:
    def __init__(self, timeout: int = 10, max_workers: int = 5):
        self.timeout = timeout
        self.max_workers = max_workers
        
        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set a user agent to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def validate_single_url(self, url: str) -> Dict:
        """Validate a single URL and return status information"""
        if not url or pd.isna(url):
            return {
                'url': url,
                'is_valid': False,
                'status_code': None,
                'response_time': None,
                'error': 'Empty URL',
                'final_url': None
            }
        
        try:
            # Clean the URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            start_time = time.time()
            response = self.session.get(
                url, 
                timeout=self.timeout,
                allow_redirects=True
            )
            response_time = time.time() - start_time
            
            return {
                'url': url,
                'is_valid': response.status_code == 200,
                'status_code': response.status_code,
                'response_time': round(response_time, 2),
                'error': None,
                'final_url': response.url if response.url != url else None
            }
            
        except requests.exceptions.Timeout:
            return {
                'url': url,
                'is_valid': False,
                'status_code': None,
                'response_time': None,
                'error': 'Timeout',
                'final_url': None
            }
        except requests.exceptions.ConnectionError:
            return {
                'url': url,
                'is_valid': False,
                'status_code': None,
                'response_time': None,
                'error': 'Connection Error',
                'final_url': None
            }
        except requests.exceptions.RequestException as e:
            return {
                'url': url,
                'is_valid': False,
                'status_code': None,
                'response_time': None,
                'error': str(e),
                'final_url': None
            }
    
    def validate_urls_batch(self, urls: list) -> Dict[str, Dict]:
        """Validate multiple URLs concurrently"""
        results = {}
        
        # Filter out empty URLs
        valid_urls = [url for url in urls if url and pd.notna(url)]
        
        logger.info(f"Validating {len(valid_urls)} URLs...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all URLs for validation
            future_to_url = {
                executor.submit(self.validate_single_url, url): url 
                for url in valid_urls
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results[url] = result
                    
                    if result['is_valid']:
                        logger.info(f"✓ {url} - Valid ({result['status_code']}) {result['response_time']}s")
                    else:
                        logger.warning(f"✗ {url} - Invalid: {result['error'] or result['status_code']}")
                        
                except Exception as e:
                    logger.error(f"Error validating {url}: {e}")
                    results[url] = {
                        'url': url,
                        'is_valid': False,
                        'status_code': None,
                        'response_time': None,
                        'error': str(e),
                        'final_url': None
                    }
        
        return results
    
    def validate_csv_websites(self, csv_file: str, output_file: str):
        """Validate all websites in the CSV and add validation columns"""
        logger.info(f"Loading data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Collect all unique URLs
        consignor_urls = df['consignor_website'].dropna().unique().tolist()
        buyer_urls = df['buyer_website'].dropna().unique().tolist()
        
        all_urls = list(set(consignor_urls + buyer_urls))
        
        logger.info(f"Found {len(all_urls)} unique URLs to validate")
        
        # Validate all URLs
        validation_results = self.validate_urls_batch(all_urls)
        
        # Add validation columns to dataframe
        df['consignor_website_valid'] = df['consignor_website'].map(
            lambda x: validation_results.get(x, {}).get('is_valid', False) if pd.notna(x) else False
        )
        df['consignor_website_status'] = df['consignor_website'].map(
            lambda x: validation_results.get(x, {}).get('status_code') if pd.notna(x) else None
        )
        df['consignor_website_error'] = df['consignor_website'].map(
            lambda x: validation_results.get(x, {}).get('error') if pd.notna(x) else None
        )
        
        df['buyer_website_valid'] = df['buyer_website'].map(
            lambda x: validation_results.get(x, {}).get('is_valid', False) if pd.notna(x) else False
        )
        df['buyer_website_status'] = df['buyer_website'].map(
            lambda x: validation_results.get(x, {}).get('status_code') if pd.notna(x) else None
        )
        df['buyer_website_error'] = df['buyer_website'].map(
            lambda x: validation_results.get(x, {}).get('error') if pd.notna(x) else None
        )
        
        # Save results
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Print validation statistics
        total_consignor_websites = df['consignor_website'].notna().sum()
        valid_consignor_websites = df['consignor_website_valid'].sum()
        total_buyer_websites = df['buyer_website'].notna().sum()
        valid_buyer_websites = df['buyer_website_valid'].sum()
        
        logger.info(f"\nValidation Results:")
        logger.info(f"Consignor websites: {valid_consignor_websites}/{total_consignor_websites} valid ({valid_consignor_websites/total_consignor_websites*100:.1f}%)" if total_consignor_websites > 0 else "No consignor websites found")
        logger.info(f"Buyer websites: {valid_buyer_websites}/{total_buyer_websites} valid ({valid_buyer_websites/total_buyer_websites*100:.1f}%)" if total_buyer_websites > 0 else "No buyer websites found")
        
        # Show some examples of invalid websites
        invalid_consignor = df[
            (df['consignor_website'].notna()) & 
            (~df['consignor_website_valid'])
        ][['consignor_cleaned', 'consignor_website', 'consignor_website_error']].drop_duplicates().head(5)
        
        if not invalid_consignor.empty:
            logger.info(f"\nSample invalid consignor websites:")
            for _, row in invalid_consignor.iterrows():
                logger.info(f"  {row['consignor_cleaned']}: {row['consignor_website']} - {row['consignor_website_error']}")
        
        invalid_buyer = df[
            (df['buyer_website'].notna()) & 
            (~df['buyer_website_valid'])
        ][['buyer_cleaned', 'buyer_website', 'buyer_website_error']].drop_duplicates().head(5)
        
        if not invalid_buyer.empty:
            logger.info(f"\nSample invalid buyer websites:")
            for _, row in invalid_buyer.iterrows():
                logger.info(f"  {row['buyer_cleaned']}: {row['buyer_website']} - {row['buyer_website_error']}")

if __name__ == "__main__":
    validator = WebsiteValidator(timeout=10, max_workers=5)
    
    # Validate websites in the enriched CSV
    validator.validate_csv_websites(
        csv_file='/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_enriched_v2.csv',
        output_file='/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_validated.csv'
    )