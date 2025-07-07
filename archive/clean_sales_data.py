import pandas as pd
import re
import json
import requests
from typing import Dict, Optional, Set
import time
from collections import defaultdict

class SalesDataCleaner:
    def __init__(self):
        # Memory dictionaries to avoid duplicate API calls
        self.consignor_mapping = {}
        self.buyer_mapping = {}
        self.contact_cache = {}
        
        # Track processed names to avoid duplicates
        self.processed_consignors = set()
        self.processed_buyers = set()
        
        # Similar name groups for deduplication
        self.consignor_groups = defaultdict(list)
        self.buyer_groups = defaultdict(list)
    
    def clean_agent_text(self, text: str, entity_type: str) -> str:
        """Clean agent text from consignor/buyer names using simple rules"""
        if pd.isna(text) or text.strip() == "":
            return None
            
        text = text.strip()
        
        # Handle special buyer cases
        if entity_type == "buyer":
            if text.lower() in ["out", "not sold", "withdrawn"] or "r.n.a" in text.lower():
                return None
        
        # Remove common agent-related terms
        agent_patterns = [
            r'\s+agent\s*[ivx]*\s*$',
            r'\s+agent\s+for\s+.*$',
            r'\s+agent\s*$',
            r'\s*,\s*agent.*$',
            r'\s*\(.*agent.*\).*$'
        ]
        
        cleaned = text
        for pattern in agent_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra spaces and formatting
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = re.sub(r'^the\s+', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned if cleaned else None
    
    def normalize_name(self, name: str) -> str:
        """Create a normalized version for deduplication"""
        if not name:
            return ""
        
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^\w\s]', '', name.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove common business suffixes
        suffixes = ['llc', 'inc', 'corp', 'ltd', 'farm', 'farms', 'ranch', 'stables', 'stable']
        words = normalized.split()
        words = [w for w in words if w not in suffixes]
        
        return ' '.join(words)
    
    def find_similar_group(self, name: str, groups: Dict, threshold: float = 0.8) -> Optional[str]:
        """Find if a name belongs to an existing similar group"""
        normalized = self.normalize_name(name)
        
        for group_key, group_names in groups.items():
            group_normalized = self.normalize_name(group_key)
            
            # Simple similarity check - could be enhanced with more sophisticated matching
            if self.simple_similarity(normalized, group_normalized) > threshold:
                return group_key
                
        return None
    
    def simple_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple similarity between two strings"""
        if not str1 or not str2:
            return 0.0
        
        # Jaccard similarity using word sets
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def deduplicate_name(self, name: str, entity_type: str) -> str:
        """Deduplicate names using similarity groups"""
        if not name:
            return name
            
        groups = self.consignor_groups if entity_type == "consignor" else self.buyer_groups
        mapping = self.consignor_mapping if entity_type == "consignor" else self.buyer_mapping
        
        # Check if we've already processed this exact name
        if name in mapping:
            return mapping[name]
        
        # Find similar group
        similar_group = self.find_similar_group(name, groups)
        
        if similar_group:
            # Use the canonical name from the group
            canonical_name = similar_group
            groups[similar_group].append(name)
        else:
            # Create new group with this name as canonical
            canonical_name = name
            groups[name] = [name]
        
        # Cache the mapping
        mapping[name] = canonical_name
        return canonical_name
    
    def extract_contact_info(self, text: str) -> Dict:
        """Extract contact information using regex patterns"""
        if not text or text.lower() == "null":
            return {"websites": [], "emails": [], "phones": []}
        
        # Extract websites
        website_pattern = r'https?://[^\s"\',<>]+'
        websites = re.findall(website_pattern, text)
        
        # Extract emails
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        
        # Extract phone numbers
        phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        
        return {
            "websites": websites,
            "emails": emails,
            "phones": phones
        }
    
    def extract_location_info(self, city_text: str, state_text: str, country_text: str) -> Dict:
        """Extract and normalize location information"""
        city = city_text.strip() if city_text and city_text.lower() != "null" else None
        
        # Extract state abbreviation
        state = None
        if state_text and state_text.lower() != "null":
            state_match = re.search(r'\b([A-Z]{2})\b', state_text.upper())
            if state_match:
                state = state_match.group(1)
        
        # Extract country
        country = None
        if country_text and country_text.lower() != "null":
            country_match = re.search(r'\b([A-Z]{2,})\b', country_text.upper())
            if country_match:
                country = country_match.group(1)
        
        return {
            "city": city,
            "state": state,
            "country": country
        }
    
    def determine_entity_type(self, name: str) -> Optional[bool]:
        """Determine if entity is a company (True) or individual (False)"""
        if not name:
            return None
        
        # Simple rules-based approach
        company_indicators = [
            'llc', 'inc', 'corp', 'ltd', 'farm', 'farms', 'ranch', 'stables', 
            'stable', 'bloodstock', 'thoroughbreds', 'equine', 'racing'
        ]
        
        name_lower = name.lower()
        for indicator in company_indicators:
            if indicator in name_lower:
                return True
        
        # If it has multiple words and no company indicators, likely individual
        words = name.split()
        if len(words) >= 2 and len(words) <= 4:
            return False
        
        return None  # Uncertain
    
    def process_sales_data(self, csv_file: str) -> pd.DataFrame:
        """Process the entire sales dataset"""
        print("Loading sales data...")
        df = pd.read_csv(csv_file)
        
        print(f"Processing {len(df)} records...")
        
        # Clean and deduplicate consignors
        print("Cleaning consignors...")
        df['consignor_cleaned'] = df['consignor'].apply(
            lambda x: self.clean_agent_text(x, "consignor")
        )
        df['consignor_cleaned'] = df['consignor_cleaned'].apply(
            lambda x: self.deduplicate_name(x, "consignor") if x else x
        )
        
        # Clean and deduplicate buyers
        print("Cleaning buyers...")
        df['buyer_cleaned'] = df['buyer'].apply(
            lambda x: self.clean_agent_text(x, "buyer")
        )
        df['buyer_cleaned'] = df['buyer_cleaned'].apply(
            lambda x: self.deduplicate_name(x, "buyer") if x else x
        )
        
        # Add entity type detection
        print("Determining entity types...")
        df['consignor_is_company'] = df['consignor_cleaned'].apply(self.determine_entity_type)
        df['buyer_is_company'] = df['buyer_cleaned'].apply(self.determine_entity_type)
        
        # Placeholder columns for contact info (would need external API integration)
        df['consignor_website'] = None
        df['consignor_email'] = None
        df['consignor_phone'] = None
        df['consignor_city'] = None
        df['consignor_state'] = None
        df['consignor_country'] = None
        
        df['buyer_website'] = None
        df['buyer_email'] = None
        df['buyer_phone'] = None
        df['buyer_city'] = None
        df['buyer_state'] = None
        df['buyer_country'] = None
        
        print(f"Cleaning complete!")
        print(f"Unique consignors: {df['consignor_cleaned'].nunique()}")
        print(f"Unique buyers: {df['buyer_cleaned'].nunique()}")
        
        return df
    
    def save_results(self, df: pd.DataFrame, output_file: str):
        """Save the cleaned results"""
        # Reorder columns
        columns = [
            'buyer', 'buyer_cleaned', 'consignor', 'consignor_cleaned',
            'sale_price', 'sale_year', 'source',
            'consignor_website', 'consignor_email', 'consignor_phone',
            'consignor_city', 'consignor_state', 'consignor_country', 'consignor_is_company',
            'buyer_website', 'buyer_email', 'buyer_phone',
            'buyer_city', 'buyer_state', 'buyer_country', 'buyer_is_company'
        ]
        
        df_output = df[columns]
        df_output.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

# Main execution
if __name__ == "__main__":
    cleaner = SalesDataCleaner()
    
    # Process the data
    df_cleaned = cleaner.process_sales_data('/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_v4.csv')
    
    # Save results
    cleaner.save_results(df_cleaned, '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_cleaned.csv')
    
    # Print some statistics
    print("\nCleaning Statistics:")
    print(f"Original consignors: {df_cleaned['consignor'].nunique()}")
    print(f"Cleaned consignors: {df_cleaned['consignor_cleaned'].nunique()}")
    print(f"Original buyers: {df_cleaned['buyer'].nunique()}")
    print(f"Cleaned buyers: {df_cleaned['buyer_cleaned'].nunique()}")
    
    print(f"\nConsignor deduplication groups: {len(cleaner.consignor_groups)}")
    print(f"Buyer deduplication groups: {len(cleaner.buyer_groups)}")