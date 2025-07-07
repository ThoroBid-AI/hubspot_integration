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

class ContactInfoEnricher:
    def __init__(self, api_key: str):
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Configure model with low temperature to reduce hallucination
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
        
        # Rate limiting
        self.last_api_call = 0
        self.min_interval = 1.0  # Minimum seconds between API calls
    
    def rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        elapsed = current_time - self.last_api_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
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
Find the official contact information for this horse industry {entity_type}: "{entity_name}"

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
                contact_info = self.parse_contact_response(response.text)
                
                # Cache the result
                self.contact_cache[cache_key] = contact_info
                
                logger.info(f"Successfully retrieved contact info for {entity_name}")
                return contact_info
            else:
                logger.warning(f"No response received for {entity_name}")
                return self.empty_contact_info()
                
        except Exception as e:
            logger.error(f"Error getting contact info for {entity_name}: {str(e)}")
            return self.empty_contact_info()
    
    def parse_contact_response(self, response_text: str) -> Dict:
        """Parse the JSON response from Gemini"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                contact_data = json.loads(json_str)
                
                # Clean and validate the data
                return {
                    'website': self.clean_website(contact_data.get('website')),
                    'email': self.clean_email(contact_data.get('email')),
                    'phone': self.clean_phone(contact_data.get('phone')),
                    'city': self.clean_text(contact_data.get('city')),
                    'state': self.clean_state(contact_data.get('state')),
                    'country': self.clean_text(contact_data.get('country')),
                    'is_company': contact_data.get('is_company')
                }
            else:
                logger.warning(f"No JSON found in response: {response_text}")
                return self.empty_contact_info()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text}")
            return self.empty_contact_info()
    
    def clean_website(self, website: str) -> Optional[str]:
        """Clean and validate website URL"""
        if not website or website.lower() == "null":
            return None
        
        # Extract valid URLs
        url_pattern = r'https?://[^\s"\',<>]+'
        urls = re.findall(url_pattern, website)
        
        return urls[0] if urls else None
    
    def clean_email(self, email: str) -> Optional[str]:
        """Clean and validate email address"""
        if not email or email.lower() == "null":
            return None
        
        # Extract valid email addresses
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, email)
        
        return emails[0] if emails else None
    
    def clean_phone(self, phone: str) -> Optional[str]:
        """Clean and validate phone number"""
        if not phone or phone.lower() == "null":
            return None
        
        # Extract valid US phone numbers
        phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, phone)
        
        return phones[0] if phones else None
    
    def clean_state(self, state: str) -> Optional[str]:
        """Clean and validate US state abbreviation"""
        if not state or state.lower() == "null":
            return None
        
        # Extract 2-letter state abbreviations
        state_pattern = r'\b([A-Z]{2})\b'
        states = re.findall(state_pattern, state.upper())
        
        return states[0] if states else None
    
    def clean_text(self, text: str) -> Optional[str]:
        """Clean general text fields"""
        if not text or text.lower() == "null":
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
    
    def enrich_sales_data(self, csv_file: str, output_file: str, max_entities: int = None):
        """Enrich the sales data with contact information"""
        logger.info("Loading sales data...")
        df = pd.read_csv(csv_file)
        
        # Get unique entities to process, sorted by frequency (most common first)
        consignor_counts = df['consignor_cleaned'].value_counts()
        buyer_counts = df['buyer_cleaned'].value_counts()
        
        unique_consignors = consignor_counts.index.tolist()
        unique_buyers = buyer_counts.index.tolist()
        
        if max_entities:
            unique_consignors = unique_consignors[:max_entities]
            unique_buyers = unique_buyers[:max_entities]
        
        logger.info(f"Processing {len(unique_consignors)} unique consignors and {len(unique_buyers)} unique buyers")
        
        # Process consignors
        consignor_info = {}
        for i, consignor in enumerate(unique_consignors):
            logger.info(f"Processing consignor {i+1}/{len(unique_consignors)}: {consignor}")
            consignor_info[consignor] = self.get_contact_info(consignor, "consignor")
        
        # Process buyers
        buyer_info = {}
        for i, buyer in enumerate(unique_buyers):
            logger.info(f"Processing buyer {i+1}/{len(unique_buyers)}: {buyer}")
            buyer_info[buyer] = self.get_contact_info(buyer, "buyer")
        
        # Apply the enriched data to the dataframe
        logger.info("Applying enriched data to dataframe...")
        
        # Helper function to safely get values and handle None
        def safe_get(mapping, key, field):
            if pd.isna(key) or key not in mapping:
                return None
            return mapping[key].get(field)
        
        # Consignor fields
        df['consignor_website'] = df['consignor_cleaned'].apply(lambda x: safe_get(consignor_info, x, 'website'))
        df['consignor_email'] = df['consignor_cleaned'].apply(lambda x: safe_get(consignor_info, x, 'email'))
        df['consignor_phone'] = df['consignor_cleaned'].apply(lambda x: safe_get(consignor_info, x, 'phone'))
        df['consignor_city'] = df['consignor_cleaned'].apply(lambda x: safe_get(consignor_info, x, 'city'))
        df['consignor_state'] = df['consignor_cleaned'].apply(lambda x: safe_get(consignor_info, x, 'state'))
        df['consignor_country'] = df['consignor_cleaned'].apply(lambda x: safe_get(consignor_info, x, 'country'))
        
        # Update is_company with LLM results where available, but keep existing values if LLM returns None
        for idx, row in df.iterrows():
            consignor_name = row['consignor_cleaned']
            if pd.notna(consignor_name) and consignor_name in consignor_info:
                llm_is_company = consignor_info[consignor_name].get('is_company')
                if llm_is_company is not None:
                    df.at[idx, 'consignor_is_company'] = llm_is_company
        
        # Buyer fields
        df['buyer_website'] = df['buyer_cleaned'].apply(lambda x: safe_get(buyer_info, x, 'website'))
        df['buyer_email'] = df['buyer_cleaned'].apply(lambda x: safe_get(buyer_info, x, 'email'))
        df['buyer_phone'] = df['buyer_cleaned'].apply(lambda x: safe_get(buyer_info, x, 'phone'))
        df['buyer_city'] = df['buyer_cleaned'].apply(lambda x: safe_get(buyer_info, x, 'city'))
        df['buyer_state'] = df['buyer_cleaned'].apply(lambda x: safe_get(buyer_info, x, 'state'))
        df['buyer_country'] = df['buyer_cleaned'].apply(lambda x: safe_get(buyer_info, x, 'country'))
        
        # Update is_company with LLM results where available, but keep existing values if LLM returns None
        for idx, row in df.iterrows():
            buyer_name = row['buyer_cleaned']
            if pd.notna(buyer_name) and buyer_name in buyer_info:
                llm_is_company = buyer_info[buyer_name].get('is_company')
                if llm_is_company is not None:
                    df.at[idx, 'buyer_is_company'] = llm_is_company
        
        # Save the enriched data
        df.to_csv(output_file, index=False)
        logger.info(f"Enriched data saved to: {output_file}")
        
        # Print statistics
        logger.info(f"\nEnrichment Statistics:")
        logger.info(f"Consignors with websites: {df['consignor_website'].notna().sum()}")
        logger.info(f"Consignors with emails: {df['consignor_email'].notna().sum()}")
        logger.info(f"Consignors with phones: {df['consignor_phone'].notna().sum()}")
        logger.info(f"Buyers with websites: {df['buyer_website'].notna().sum()}")
        logger.info(f"Buyers with emails: {df['buyer_email'].notna().sum()}")
        logger.info(f"Buyers with phones: {df['buyer_phone'].notna().sum()}")
        
        # Debug: Show some sample enriched data
        logger.info(f"\nSample enriched consignors:")
        for name, info in list(consignor_info.items())[:3]:
            logger.info(f"  {name}: {info}")
        
        logger.info(f"\nSample enriched buyers:")
        for name, info in list(buyer_info.items())[:3]:
            logger.info(f"  {name}: {info}")

# Main execution
if __name__ == "__main__":
    # Set API key
    api_key = "AIzaSyBB2-SGq0Pjbp4QDEBLVcejzz0LWg1O3_Q"
    
    enricher = ContactInfoEnricher(api_key)
    
    # Process the cleaned data with better-known entities for testing
    enricher.enrich_sales_data(
        csv_file='/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_cleaned.csv',
        output_file='/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_enriched_v2.csv',
        max_entities=10  # Test with top 10 entities
    )