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
        
        # Use the regular model instead of experimental
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-preview',  # Use preview instead of exp for higher rate limits
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temperature for factual accuracy
                top_p=0.8,
                top_k=20,
                max_output_tokens=500
            )
        )
        
        # Cache to avoid duplicate API calls
        self.contact_cache = {}
        
        # Rate limiting - more conservative
        self.last_api_call = 0
        self.min_interval = 6.0  # 6 seconds between calls to stay under rate limit
    
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
            if "429" in str(e):  # Rate limit error
                logger.info("Rate limit hit, sleeping for 60 seconds...")
                time.sleep(60)
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

# Example usage for batch processing with rate limiting
def process_entities_batch(entities: List[str], entity_type: str, api_key: str, batch_size: int = 50):
    """Process entities in batches to handle rate limits"""
    enricher = ContactInfoEnricher(api_key)
    
    results = {}
    total_entities = len(entities)
    
    for i in range(0, total_entities, batch_size):
        batch = entities[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} entities)")
        
        for j, entity in enumerate(batch):
            try:
                logger.info(f"Processing {entity_type} {i+j+1}/{total_entities}: {entity}")
                results[entity] = enricher.get_contact_info(entity, entity_type)
            except Exception as e:
                logger.error(f"Error processing {entity}: {e}")
                results[entity] = enricher.empty_contact_info()
        
        # Longer break between batches
        if i + batch_size < total_entities:
            logger.info("Batch complete, waiting 2 minutes before next batch...")
            time.sleep(120)
    
    return results

if __name__ == "__main__":
    # Example: Process a few entities
    api_key = "AIzaSyBB2-SGq0Pjbp4QDEBLVcejzz0LWg1O3_Q"
    
    # Test with a small batch
    test_consignors = [
        "Taylor Made Sales Agency",
        "Claiborne Farm", 
        "Three Chimneys Farm",
        "Lane's End Farm",
        "WinStar Farm"
    ]
    
    logger.info("Processing test batch of well-known consignors...")
    results = process_entities_batch(test_consignors, "consignor", api_key, batch_size=5)
    
    # Print results
    for entity, info in results.items():
        logger.info(f"\n{entity}:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")