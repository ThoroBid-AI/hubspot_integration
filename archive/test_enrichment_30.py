import pandas as pd
import google.generativeai as genai
import os
import time
import re
import json
from typing import Dict, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestContactEnricher:
    def __init__(self, api_key: str):
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Use the working model
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temperature for factual accuracy
                top_p=0.8,
                top_k=20,
                max_output_tokens=500
            )
        )
        
        # Cache to avoid duplicate API calls
        self.contact_cache = {}
        
        # Rate limiting - very conservative for testing
        self.last_api_call = 0
        self.min_interval = 7.0  # 7 seconds between calls to avoid hitting limits
    
    def rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        elapsed = current_time - self.last_api_call
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        self.last_api_call = time.time()
    
    def get_contact_info(self, entity_name: str, entity_type: str) -> Dict:
        """Get contact information for an entity using Gemini with web search"""
        if not entity_name or entity_name.strip() == "":
            return self.empty_contact_info()
        
        # Check cache first
        cache_key = f"{entity_type}:{entity_name.lower()}"
        if cache_key in self.contact_cache:
            logger.info(f"Using cached data for {entity_name}")
            return self.contact_cache[cache_key]
        
        self.rate_limit()
        
        try:
            # Prompt for getting contact information
            prompt = f"""
Search the web and find the official contact information for this horse industry {entity_type}: "{entity_name}"

Please provide ONLY the following information in this exact JSON format:
{{
    "website": "official website URL or null",
    "email": "official email address or null", 
    "phone": "official phone number or null",
    "city": "city name or null",
    "state": "US state abbreviation (e.g., KY, FL) or null",
    "country": "country name or null",
    "is_company": true/false
}}

Rules:
- Return "null" (not empty string) if information not found
- Use web search to find current, accurate information
- Only return official/verified contact details
- For phone numbers, use US format if available
- For states, use 2-letter abbreviations (KY, FL, CA, etc.)
- is_company should be true for businesses/farms/stables, false for individuals
- Do not hallucinate or guess information
"""
            
            logger.info(f"Querying contact info for: {entity_name}")
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                # Try to parse JSON response
                contact_info = self.parse_contact_response(response.text, entity_name)
                
                # Cache the result
                self.contact_cache[cache_key] = contact_info
                
                logger.info(f"Successfully retrieved contact info for {entity_name}")
                return contact_info
            else:
                logger.warning(f"No response received for {entity_name}")
                return self.empty_contact_info()
                
        except Exception as e:
            logger.error(f"Error getting contact info for {entity_name}: {str(e)}")
            if "429" in str(e):  # Rate limit error
                logger.info("Rate limit hit, sleeping for 60 seconds...")
                time.sleep(60)
            return self.empty_contact_info()
    
    def parse_contact_response(self, response_text: str, entity_name: str) -> Dict:
        """Parse the JSON response from Gemini"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                contact_data = json.loads(json_str)
                
                # Clean and validate the data
                result = {
                    'website': self.clean_website(contact_data.get('website')),
                    'email': self.clean_email(contact_data.get('email')),
                    'phone': self.clean_phone(contact_data.get('phone')),
                    'city': self.clean_text(contact_data.get('city')),
                    'state': self.clean_state(contact_data.get('state')),
                    'country': self.clean_text(contact_data.get('country')),
                    'is_company': contact_data.get('is_company')
                }
                
                # Log the results for visibility
                logger.info(f"  Results for {entity_name}:")
                for key, value in result.items():
                    if value:
                        logger.info(f"    {key}: {value}")
                
                return result
            else:
                logger.warning(f"No JSON found in response for {entity_name}: {response_text[:200]}")
                return self.empty_contact_info()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {entity_name}: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            return self.empty_contact_info()
    
    def clean_website(self, website: str) -> Optional[str]:
        """Clean and validate website URL"""
        if not website or website == "null":
            return None
        
        # Extract valid URLs
        url_pattern = r'https?://[^\s"\',<>]+'
        urls = re.findall(url_pattern, website)
        
        return urls[0] if urls else None
    
    def clean_email(self, email: str) -> Optional[str]:
        """Clean and validate email address"""
        if not email or email == "null":
            return None
        
        # Extract valid email addresses
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, email)
        
        return emails[0] if emails else None
    
    def clean_phone(self, phone: str) -> Optional[str]:
        """Clean and validate phone number"""
        if not phone or phone == "null":
            return None
        
        # Extract valid US phone numbers
        phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, phone)
        
        return phones[0] if phones else None
    
    def clean_state(self, state: str) -> Optional[str]:
        """Clean and validate US state abbreviation"""
        if not state or state == "null":
            return None
        
        # Extract 2-letter state abbreviations
        state_pattern = r'\b([A-Z]{2})\b'
        states = re.findall(state_pattern, state.upper())
        
        return states[0] if states else None
    
    def clean_text(self, text: str) -> Optional[str]:
        """Clean general text fields"""
        if not text or text == "null":
            return None
        
        cleaned = text.strip()
        return cleaned if cleaned else None
    
    def empty_contact_info(self) -> Dict:
        """Return empty contact info structure"""
        return {
            'website': None,
            'email': None,
            'phone': None,
            'city': None,
            'state': None,
            'country': None,
            'is_company': None
        }

def run_test_enrichment():
    """Run enrichment test on 30 entities (15 consignors + 15 buyers)"""
    
    # Load the final dataset
    logger.info("Loading dataset to select test entities...")
    df = pd.read_csv('/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_enriched.csv')
    
    # Get entities from past 3 years for more relevance
    recent_data = df[df['sale_year'].isin([2023, 2024, 2025])]
    
    # Select top 15 consignors by frequency (most active)
    top_consignors = recent_data['consignor_cleaned'].value_counts().head(15).index.tolist()
    
    # Select top 15 buyers by frequency (excluding non-buyers)
    actual_buyers = recent_data[
        recent_data['buyer_cleaned'].notna() & 
        ~recent_data['buyer_cleaned'].isin(['NOT SOLD', 'OUT', 'WITHDRAWN', '', 'RNA'])
    ]
    top_buyers = actual_buyers['buyer_cleaned'].value_counts().head(15).index.tolist()
    
    logger.info(f"Selected test entities:")
    logger.info(f"  Top 15 consignors: {top_consignors}")
    logger.info(f"  Top 15 buyers: {top_buyers}")
    
    # Initialize enricher
    api_key = "AIzaSyBB2-SGq0Pjbp4QDEBLVcejzz0LWg1O3_Q"
    enricher = TestContactEnricher(api_key)
    
    # Store results
    consignor_results = {}
    buyer_results = {}
    
    print(f"\n{'='*80}")
    print(f"STARTING TEST ENRICHMENT - 30 ENTITIES")
    print(f"{'='*80}")
    
    # Process consignors
    print(f"\n{'='*60}")
    print(f"PROCESSING 15 CONSIGNORS")
    print(f"{'='*60}")
    
    for i, consignor in enumerate(top_consignors, 1):
        print(f"\n[{i}/15] Processing consignor: {consignor}")
        try:
            result = enricher.get_contact_info(consignor, "consignor")
            consignor_results[consignor] = result
        except Exception as e:
            logger.error(f"Error processing {consignor}: {e}")
            consignor_results[consignor] = enricher.empty_contact_info()
    
    # Process buyers
    print(f"\n{'='*60}")
    print(f"PROCESSING 15 BUYERS")
    print(f"{'='*60}")
    
    for i, buyer in enumerate(top_buyers, 1):
        print(f"\n[{i}/15] Processing buyer: {buyer}")
        try:
            result = enricher.get_contact_info(buyer, "buyer")
            buyer_results[buyer] = result
        except Exception as e:
            logger.error(f"Error processing {buyer}: {e}")
            buyer_results[buyer] = enricher.empty_contact_info()
    
    # Generate summary report
    print(f"\n{'='*80}")
    print(f"TEST ENRICHMENT RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Consignor results summary
    consignor_websites = sum(1 for r in consignor_results.values() if r['website'])
    consignor_emails = sum(1 for r in consignor_results.values() if r['email'])
    consignor_phones = sum(1 for r in consignor_results.values() if r['phone'])
    consignor_locations = sum(1 for r in consignor_results.values() if r['city'])
    
    print(f"\nConsignor Results (15 entities):")
    print(f"  Websites found: {consignor_websites}/15 ({consignor_websites/15*100:.1f}%)")
    print(f"  Emails found: {consignor_emails}/15 ({consignor_emails/15*100:.1f}%)")
    print(f"  Phones found: {consignor_phones}/15 ({consignor_phones/15*100:.1f}%)")
    print(f"  Locations found: {consignor_locations}/15 ({consignor_locations/15*100:.1f}%)")
    
    # Buyer results summary
    buyer_websites = sum(1 for r in buyer_results.values() if r['website'])
    buyer_emails = sum(1 for r in buyer_results.values() if r['email'])
    buyer_phones = sum(1 for r in buyer_results.values() if r['phone'])
    buyer_locations = sum(1 for r in buyer_results.values() if r['city'])
    
    print(f"\nBuyer Results (15 entities):")
    print(f"  Websites found: {buyer_websites}/15 ({buyer_websites/15*100:.1f}%)")
    print(f"  Emails found: {buyer_emails}/15 ({buyer_emails/15*100:.1f}%)")
    print(f"  Phones found: {buyer_phones}/15 ({buyer_phones/15*100:.1f}%)")
    print(f"  Locations found: {buyer_locations}/15 ({buyer_locations/15*100:.1f}%)")
    
    # Show successful examples
    print(f"\n{'='*60}")
    print(f"SUCCESSFUL ENRICHMENT EXAMPLES")
    print(f"{'='*60}")
    
    print(f"\nConsignors with complete contact info:")
    for name, info in consignor_results.items():
        if info['website'] or info['email'] or info['phone']:
            print(f"\n  {name}:")
            for key, value in info.items():
                if value:
                    print(f"    {key}: {value}")
    
    print(f"\nBuyers with complete contact info:")
    for name, info in buyer_results.items():
        if info['website'] or info['email'] or info['phone']:
            print(f"\n  {name}:")
            for key, value in info.items():
                if value:
                    print(f"    {key}: {value}")
    
    # Save detailed results
    results_file = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/test_enrichment_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'consignors': consignor_results,
            'buyers': buyer_results,
            'summary': {
                'consignor_success_rate': {
                    'websites': f"{consignor_websites}/15",
                    'emails': f"{consignor_emails}/15",
                    'phones': f"{consignor_phones}/15",
                    'locations': f"{consignor_locations}/15"
                },
                'buyer_success_rate': {
                    'websites': f"{buyer_websites}/15",
                    'emails': f"{buyer_emails}/15",
                    'phones': f"{buyer_phones}/15",
                    'locations': f"{buyer_locations}/15"
                }
            }
        }, indent=2)
    
    print(f"\n{'='*80}")
    print(f"TEST COMPLETE")
    print(f"{'='*80}")
    print(f"Detailed results saved to: test_enrichment_results.json")
    print(f"Total entities processed: 30 (15 consignors + 15 buyers)")
    print(f"Overall success rate: {(consignor_websites + consignor_emails + consignor_phones + buyer_websites + buyer_emails + buyer_phones)} contact points found")

if __name__ == "__main__":
    run_test_enrichment()