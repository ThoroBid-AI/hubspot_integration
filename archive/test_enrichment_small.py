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

class SmallTestEnricher:
    def __init__(self, api_key: str):
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Use the working model
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=20,
                max_output_tokens=500
            )
        )
        
        # Rate limiting - conservative
        self.last_api_call = 0
        self.min_interval = 8.0  # 8 seconds between calls
    
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
        """Get contact information for an entity"""
        self.rate_limit()
        
        try:
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
                return self.parse_response(response.text, entity_name)
            else:
                logger.warning(f"No response received for {entity_name}")
                return self.empty_result()
                
        except Exception as e:
            logger.error(f"Error getting contact info for {entity_name}: {str(e)}")
            return self.empty_result()
    
    def parse_response(self, response_text: str, entity_name: str) -> Dict:
        """Parse the JSON response"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                contact_data = json.loads(json_str)
                
                result = {
                    'website': contact_data.get('website') if contact_data.get('website') != 'null' else None,
                    'email': contact_data.get('email') if contact_data.get('email') != 'null' else None,
                    'phone': contact_data.get('phone') if contact_data.get('phone') != 'null' else None,
                    'city': contact_data.get('city') if contact_data.get('city') != 'null' else None,
                    'state': contact_data.get('state') if contact_data.get('state') != 'null' else None,
                    'country': contact_data.get('country') if contact_data.get('country') != 'null' else None,
                    'is_company': contact_data.get('is_company')
                }
                
                # Show results
                logger.info(f"Results for {entity_name}:")
                for key, value in result.items():
                    if value:
                        logger.info(f"  {key}: {value}")
                
                return result
            else:
                logger.warning(f"No JSON found for {entity_name}")
                return self.empty_result()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {entity_name}: {e}")
            return self.empty_result()
    
    def empty_result(self) -> Dict:
        return {
            'website': None,
            'email': None,
            'phone': None,
            'city': None,
            'state': None,
            'country': None,
            'is_company': None
        }

def run_small_test():
    """Run enrichment test on 10 entities (5 consignors + 5 buyers)"""
    
    # Test with well-known entities that are likely to have online presence
    test_consignors = [
        "de Meric Sales",
        "Top Line Sales LLC", 
        "Niall Brennan Stables",
        "Wavertree Stables",
        "Ocala Stud"
    ]
    
    test_buyers = [
        "West Point Thoroughbreds Inc. L.E.B.",
        "Calumet Farm",
        "K.O.I.D. Co. Ltd.",
        "Starship Stables",
        "Legion Bloodstock"
    ]
    
    api_key = "AIzaSyBB2-SGq0Pjbp4QDEBLVcejzz0LWg1O3_Q"
    enricher = SmallTestEnricher(api_key)
    
    results = {'consignors': {}, 'buyers': {}}
    
    print(f"\n{'='*60}")
    print(f"SMALL TEST: 10 ENTITIES (5 CONSIGNORS + 5 BUYERS)")
    print(f"{'='*60}")
    
    # Process consignors
    print(f"\nProcessing 5 consignors...")
    for i, consignor in enumerate(test_consignors, 1):
        print(f"\n[{i}/5] {consignor}")
        try:
            result = enricher.get_contact_info(consignor, "consignor")
            results['consignors'][consignor] = result
        except Exception as e:
            logger.error(f"Error: {e}")
            results['consignors'][consignor] = enricher.empty_result()
    
    # Process buyers
    print(f"\nProcessing 5 buyers...")
    for i, buyer in enumerate(test_buyers, 1):
        print(f"\n[{i}/5] {buyer}")
        try:
            result = enricher.get_contact_info(buyer, "buyer")
            results['buyers'][buyer] = result
        except Exception as e:
            logger.error(f"Error: {e}")
            results['buyers'][buyer] = enricher.empty_result()
    
    # Generate summary
    print(f"\n{'='*60}")
    print(f"TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    consignor_stats = {
        'websites': sum(1 for r in results['consignors'].values() if r['website']),
        'emails': sum(1 for r in results['consignors'].values() if r['email']),
        'phones': sum(1 for r in results['consignors'].values() if r['phone']),
        'locations': sum(1 for r in results['consignors'].values() if r['city'])
    }
    
    buyer_stats = {
        'websites': sum(1 for r in results['buyers'].values() if r['website']),
        'emails': sum(1 for r in results['buyers'].values() if r['email']),
        'phones': sum(1 for r in results['buyers'].values() if r['phone']),
        'locations': sum(1 for r in results['buyers'].values() if r['city'])
    }
    
    print(f"\nConsignor Results (5 entities):")
    for metric, count in consignor_stats.items():
        print(f"  {metric.title()}: {count}/5 ({count/5*100:.0f}%)")
    
    print(f"\nBuyer Results (5 entities):")
    for metric, count in buyer_stats.items():
        print(f"  {metric.title()}: {count}/5 ({count/5*100:.0f}%)")
    
    # Show detailed results
    print(f"\n{'='*60}")
    print(f"DETAILED RESULTS")
    print(f"{'='*60}")
    
    print(f"\nConsignors with contact info:")
    for name, info in results['consignors'].items():
        if any(info.values()):
            print(f"\n  {name}:")
            for key, value in info.items():
                if value:
                    print(f"    {key}: {value}")
    
    print(f"\nBuyers with contact info:")
    for name, info in results['buyers'].items():
        if any(info.values()):
            print(f"\n  {name}:")
            for key, value in info.items():
                if value:
                    print(f"    {key}: {value}")
    
    print(f"\n{'='*60}")
    print(f"TEST COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_small_test()