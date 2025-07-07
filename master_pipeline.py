#!/usr/bin/env python3
"""
Master Horse Sales Data Pipeline
===============================
Complete end-to-end pipeline that:
1. Queries BigQuery tables for horse sales data
2. Cleans and processes the data with unions
3. Performs web search enrichment for contact information
4. Saves final results as CSV

Usage: python master_pipeline.py
"""

import pandas as pd
import google.generativeai as genai
from google.cloud import bigquery
import asyncio
import os
import time
import re
import json
from typing import Dict, Optional, List, Tuple
import logging
import pickle
from datetime import datetime
import sys
from difflib import SequenceMatcher
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup file logging
log_filename = f"master_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.info(f"Master pipeline log file created: {log_filename}")

class MasterHorseSalesPipeline:
    def __init__(self, gemini_api_key: str, bigquery_credentials_path: str):
        """Initialize the master pipeline with all necessary configurations"""
        self.gemini_api_key = gemini_api_key
        self.bigquery_credentials_path = bigquery_credentials_path
        
        # Initialize BigQuery client
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = bigquery_credentials_path
        self.bq_client = bigquery.Client()
        
        # Configure Gemini AI
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=20,
                max_output_tokens=500
            )
        )
        
        # Pipeline configuration
        self.batch_size = 3
        self.batch_delay = 120  # 2 minutes between batches
        self.call_interval = 8  # 8 seconds between API calls
        self.max_consecutive_timeouts = 5
        
        # Tracking variables
        self.consecutive_timeouts = 0
        self.total_processed = 0
        self.total_successful = 0
        self.total_failed = 0
        self.enrichment_results = {}
        self.last_api_call = 0
        
        # Data cleaning configuration
        self.similarity_threshold = 0.8
        self.agent_patterns = [
            r'\b(agent|agt|as agent|as agt)\b',
            r'\(agent.*?\)',
            r'\[agent.*?\]',
            r'\bagent for\b',
            r'\bacting as agent\b'
        ]
        
        # Business suffixes for entity type detection
        self.business_suffixes = [
            'llc', 'inc', 'corp', 'ltd', 'farm', 'farms', 'stable', 'stables', 
            'racing', 'bloodstock', 'sales', 'training', 'stud', 'ranch',
            'partnership', 'syndicate', 'group', 'company', 'co', 'enterprises'
        ]
        
        logger.info("Master pipeline initialized successfully")

    # ==================== BIGQUERY DATA EXTRACTION ====================
    
    def query_bigquery_data(self) -> pd.DataFrame:
        """Query BigQuery for horse sales data from all sources"""
        logger.info("🔍 Querying BigQuery for horse sales data...")
        
        query = """
        SELECT
          horse_name,
          hip_number,
          color,
          sex,
          sire_name,
          dam_name,
          dam_sire,
          consignor,
          buyer_name,
          sale_price,
          barn_number,
          foaling_date,
          foaling_year,
          birth_date,
          sale_title,
          session,
          reserve_price,
          private_sale_indicator,
          covering_sire,
          last_bred_date,
          under_tack_time,
          ut_distance,
          rna_indicator,
          out_indicator,
          state_foaled,
          source_file,
          sale_year,
          source
        FROM (
          -- Ocala Sales
          SELECT
            CAST(horse_name AS STRING) as horse_name,
            CAST(hip_number AS STRING) as hip_number,
            CAST(color AS STRING) as color,
            CAST(sex AS STRING) as sex,
            CAST(sire_name AS STRING) as sire_name,
            CAST(dam_name AS STRING) as dam_name,
            CAST(dam_sire AS STRING) as dam_sire,
            CAST(consignor AS STRING) as consignor,
            CAST(buyer_name AS STRING) as buyer_name,
            CAST(sale_price AS STRING) as sale_price,
            CAST(barn_number AS STRING) as barn_number,
            CAST(foaling_date AS STRING) as foaling_date,
            CAST(foaling_year AS STRING) as foaling_year,
            CAST(foaling_date AS STRING) as birth_date,
            CAST(sale_type AS STRING) as sale_title,
            NULL as session,
            CAST(reserve_price AS STRING) as reserve_price,
            CAST(private_sale_indicator AS STRING) as private_sale_indicator,
            CAST(in_foal_sire AS STRING) as covering_sire,
            CAST(last_bred AS STRING) as last_bred_date,
            CAST(under_tack_time AS STRING) as under_tack_time,
            CAST(ut_distance AS STRING) as ut_distance,
            NULL as rna_indicator,
            CAST(in_out_status AS STRING) as out_indicator,
            CAST(foaling_area AS STRING) as state_foaled,
            CAST(source_file AS STRING) as source_file,
            CAST(source_year AS STRING) as sale_year,
            'ocala' as source
          FROM `thorobid-dev.ingest.ocala_sales`

          UNION ALL

          -- Keeneland Sales
          SELECT
            CAST(name AS STRING) as horse_name,
            CAST(hip AS STRING) as hip_number,
            CAST(color AS STRING) as color,
            CAST(sex AS STRING) as sex,
            CAST(sire AS STRING) as sire_name,
            CAST(dam AS STRING) as dam_name,
            CAST(broodmare_sire AS STRING) as dam_sire,
            CAST(consignor AS STRING) as consignor,
            CAST(buyer AS STRING) as buyer_name,
            CAST(sale_price AS STRING) as sale_price,
            CAST(barn AS STRING) as barn_number,
            CAST(dob_formatted AS STRING) as foaling_date,
            CAST(yob AS STRING) as foaling_year,
            CAST(dob AS STRING) as birth_date,
            CAST(sale AS STRING) as sale_title,
            CAST(session AS STRING) as session,
            NULL as reserve_price,
            NULL as private_sale_indicator,
            CAST(covering_sire AS STRING) as covering_sire,
            CAST(last_service_date AS STRING) as last_bred_date,
            NULL as under_tack_time,
            NULL as ut_distance,
            CAST(rna_indicator AS STRING) as rna_indicator,
            CAST(out_indicator AS STRING) as out_indicator,
            CAST(state_foaled AS STRING) as state_foaled,
            CAST(pp_file_name AS STRING) as source_file,
            CAST(REGEXP_EXTRACT(sale, r'([0-9]{4})') AS STRING) as sale_year,
            'keeneland' as source
          FROM `thorobid-dev.ingest.keeneland_sales`

          UNION ALL

          -- Fasig-Tipton Sales
          SELECT
            CAST(NAME AS STRING) as horse_name,
            CAST(HIP AS STRING) as hip_number,
            CAST(COLOR AS STRING) as color,
            CAST(SEX AS STRING) as sex,
            CAST(SIRE AS STRING) as sire_name,
            CAST(DAM AS STRING) as dam_name,
            CAST(SIRE_OF_DAM AS STRING) as dam_sire,
            CAST(CONSIGNOR_NAME AS STRING) as consignor,
            CAST(PURCHASER AS STRING) as buyer_name,
            CAST(PRICE AS STRING) as sale_price,
            CAST(BARN AS STRING) as barn_number,
            NULL as foaling_date,
            CAST(YEAR_OF_BIRTH AS STRING) as foaling_year,
            NULL as birth_date,
            CAST(SALE_TITLE AS STRING) as sale_title,
            CAST(SESSION AS STRING) as session,
            NULL as reserve_price,
            CAST(PRIVATE_SALE AS STRING) as private_sale_indicator,
            CAST(COVERING_SIRE AS STRING) as covering_sire,
            CAST(COVER_DATE AS STRING) as last_bred_date,
            NULL as under_tack_time,
            NULL as ut_distance,
            NULL as rna_indicator,
            NULL as out_indicator,
            CAST(FOALED AS STRING) as state_foaled,
            NULL as source_file,
            CAST(REGEXP_EXTRACT(SALE_TITLE, r'\(([0-9]{4})\)') AS STRING) as sale_year,
            'fasigtipton' as source
          FROM `thorobid-dev.ingest.fasigtipton_sales`
        )
        ORDER BY foaling_year DESC
        """
        
        try:
            logger.info("Executing BigQuery union query across all 3 tables...")
            df = self.bq_client.query(query).to_dataframe()
            logger.info(f"✅ Retrieved {len(df)} records from BigQuery")
            logger.info(f"Data sources: {df['source'].value_counts().to_dict()}")
            return df
        
        except Exception as e:
            logger.error(f"BigQuery error: {e}")
            raise

    # ==================== DATA CLEANING AND PROCESSING ====================
    
    def clean_agent_text(self, text: str) -> str:
        """Remove agent-related text from names"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        for pattern in self.agent_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces and punctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[,\(\)\[\]]+$', '', text)
        return text.strip()

    def normalize_name(self, name: str) -> str:
        """Normalize name for better deduplication"""
        if not name or pd.isna(name):
            return ""
        
        name = str(name).upper().strip()
        
        # Remove special characters but keep spaces and apostrophes
        name = re.sub(r"[^\w\s']", ' ', name)
        
        # Normalize unicode characters
        name = unicodedata.normalize('NFKD', name)
        
        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name)
        
        return name.strip()

    def similarity_score(self, a: str, b: str) -> float:
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a, b).ratio()

    def deduplicate_name(self, name: str, existing_names: Dict[str, str]) -> str:
        """Find the best match for a name among existing names"""
        if not name:
            return name
        
        normalized = self.normalize_name(name)
        
        # Check for exact match first
        if normalized in existing_names:
            return existing_names[normalized]
        
        # Check for similar matches
        best_match = None
        best_score = 0
        
        for existing_norm, existing_orig in existing_names.items():
            score = self.similarity_score(normalized, existing_norm)
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = existing_orig
        
        if best_match:
            return best_match
        else:
            # Add new name to registry
            existing_names[normalized] = name
            return name

    def determine_entity_type(self, name: str) -> bool:
        """Determine if entity is a company (True) or individual (False)"""
        if not name or pd.isna(name):
            return None
        
        name_lower = str(name).lower()
        
        # Check for business suffixes
        for suffix in self.business_suffixes:
            if suffix in name_lower:
                return True
        
        # Check for multiple words (likely company names)
        words = name_lower.split()
        if len(words) >= 3:
            return True
        
        # Check for specific business indicators
        business_indicators = [
            'farm', 'stable', 'racing', 'bloodstock', 'training',
            'partnership', 'syndicate', 'group', 'management'
        ]
        
        for indicator in business_indicators:
            if indicator in name_lower:
                return True
        
        return False

    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information using regex patterns"""
        if not text or pd.isna(text):
            return {'email': None, 'phone': None, 'website': None}
        
        text = str(text)
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        # Phone pattern (various formats)
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phones = re.findall(phone_pattern, text)
        
        # Website pattern
        website_pattern = r'https?://[^\s]+|www\.[^\s]+'
        websites = re.findall(website_pattern, text)
        
        return {
            'email': emails[0] if emails else None,
            'phone': f"({phones[0][1]}) {phones[0][2]}-{phones[0][3]}" if phones else None,
            'website': websites[0] if websites else None
        }

    def clean_and_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process the raw BigQuery data"""
        logger.info("🧹 Starting comprehensive data cleaning...")
        
        original_count = len(df)
        
        # Remove duplicates
        df_clean = df.drop_duplicates()
        after_dedup = len(df_clean)
        logger.info(f"Removed {original_count - after_dedup} duplicate records")
        
        # Clean agent text from names and remove commas
        df_clean['consignor_cleaned'] = df_clean['consignor'].apply(self.clean_agent_text).str.replace(',', '', regex=False)
        df_clean['buyer_cleaned'] = df_clean['buyer_name'].apply(self.clean_agent_text).str.replace(',', '', regex=False)
        
        # Remove common non-buyer values
        non_buyers = ['NOT SOLD', 'OUT', 'WITHDRAWN', 'RNA', 'PASSED', 'BUY BACK', '']
        df_clean = df_clean[~df_clean['buyer_cleaned'].isin(non_buyers)]
        
        # Remove empty entries
        df_clean = df_clean[
            (df_clean['consignor_cleaned'] != '') & 
            (df_clean['buyer_cleaned'] != '') &
            (df_clean['consignor_cleaned'].notna()) &
            (df_clean['buyer_cleaned'].notna())
        ]
        
        after_cleaning = len(df_clean)
        logger.info(f"After basic cleaning: {original_count} → {after_cleaning} records")
        
        # Deduplicate names
        logger.info("🔄 Deduplicating entity names...")
        consignor_registry = {}
        buyer_registry = {}
        
        df_clean['consignor_cleaned'] = df_clean['consignor_cleaned'].apply(
            lambda x: self.deduplicate_name(x, consignor_registry)
        )
        df_clean['buyer_cleaned'] = df_clean['buyer_cleaned'].apply(
            lambda x: self.deduplicate_name(x, buyer_registry)
        )
        
        # Determine entity types
        logger.info("🏢 Determining entity types...")
        df_clean['consignor_is_company'] = df_clean['consignor_cleaned'].apply(self.determine_entity_type)
        df_clean['buyer_is_company'] = df_clean['buyer_cleaned'].apply(self.determine_entity_type)
        
        # Initialize contact information columns
        contact_columns = [
            'consignor_website', 'consignor_email', 'consignor_phone', 
            'consignor_city', 'consignor_state', 'consignor_country',
            'buyer_website', 'buyer_email', 'buyer_phone',
            'buyer_city', 'buyer_state', 'buyer_country'
        ]
        
        for col in contact_columns:
            df_clean[col] = None
        
        final_count = len(df_clean)
        logger.info(f"✅ Data cleaning complete: {original_count} → {final_count} records ({(final_count/original_count)*100:.1f}% retained)")
        logger.info(f"Unique consignors: {df_clean['consignor_cleaned'].nunique()}")
        logger.info(f"Unique buyers: {df_clean['buyer_cleaned'].nunique()}")
        
        return df_clean

    # ==================== ENTITY SELECTION ====================
    
    def get_all_entities(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Get ALL unique entities without filtering"""
        logger.info("📊 Getting ALL unique entities...")
        
        # Get all unique consignors
        all_consignors = df['consignor_cleaned'].dropna().unique().tolist()
        
        # Get all unique buyers
        all_buyers = df['buyer_cleaned'].dropna().unique().tolist()
        
        logger.info(f"Found {len(all_consignors)} unique consignors")
        logger.info(f"Found {len(all_buyers)} unique buyers")
        logger.info(f"Total entities to enrich: {len(all_consignors) + len(all_buyers)}")
        
        return all_consignors, all_buyers

    # ==================== WEB SEARCH ENRICHMENT ====================
    
    def rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        elapsed = current_time - self.last_api_call
        if elapsed < self.call_interval:
            sleep_time = self.call_interval - elapsed
            logger.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        self.last_api_call = time.time()

    def get_contact_info(self, entity_name: str, entity_type: str) -> Dict:
        """Get contact information for a single entity using AI web search"""
        if not entity_name or entity_name.strip() == "":
            return self.empty_contact_result()

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

            logger.info(f"🔍 Querying contact info for: {entity_name}")
            response = self.model.generate_content(prompt)

            if response and response.text:
                result = self.parse_ai_response(response.text, entity_name)
                
                # Reset consecutive timeouts on success
                self.consecutive_timeouts = 0
                self.total_processed += 1
                self.total_successful += 1
                
                logger.info(f"✓ Successfully processed: {entity_name} (Processed: {self.total_processed}, Success: {self.total_successful}, Failed: {self.total_failed})")
                return result
            else:
                logger.warning(f"No response received for {entity_name}")
                self.total_processed += 1
                self.total_failed += 1
                return self.empty_contact_result()

        except Exception as e:
            logger.error(f"Error getting contact info for {entity_name}: {str(e)}")
            
            # Check for timeout/API errors
            is_timeout = any(keyword in str(e).lower() for keyword in ['timeout', '429', 'rate limit', 'quota', 'api'])
            
            if is_timeout:
                self.consecutive_timeouts += 1
                logger.warning(f"API timeout/error #{self.consecutive_timeouts} for {entity_name}")
                
                # Check if we've hit the consecutive timeout limit
                if self.consecutive_timeouts >= self.max_consecutive_timeouts:
                    logger.critical(f"CRITICAL: {self.consecutive_timeouts} consecutive API timeouts detected. Exiting process.")
                    logger.critical(f"Final stats - Processed: {self.total_processed}, Successful: {self.total_successful}, Failed: {self.total_failed}")
                    return None  # Signal to stop processing
                
                logger.info(f"Timeout detected, sleeping for 60 seconds... ({self.consecutive_timeouts}/{self.max_consecutive_timeouts} consecutive)")
                time.sleep(60)
            else:
                # Reset consecutive timeouts for non-timeout errors
                self.consecutive_timeouts = 0
            
            self.total_processed += 1
            self.total_failed += 1
            return self.empty_contact_result()

    def parse_ai_response(self, response_text: str, entity_name: str) -> Dict:
        """Parse the AI JSON response"""
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
                
                # Log successful findings
                found_items = [k for k, v in result.items() if v and k != 'is_company']
                if found_items:
                    logger.info(f"  Found: {', '.join(found_items)}")
                
                return result
            else:
                logger.warning(f"No JSON found for {entity_name}")
                return self.empty_contact_result()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {entity_name}: {e}")
            return self.empty_contact_result()

    def empty_contact_result(self) -> Dict:
        """Return empty contact information structure"""
        return {
            'website': None,
            'email': None,
            'phone': None,
            'city': None,
            'state': None,
            'country': None,
            'is_company': None
        }

    async def enrich_entities(self, consignors: List[str], buyers: List[str]) -> Dict:
        """Enrich all entities with contact information"""
        logger.info("🚀 Starting entity enrichment...")
        
        # Combine all entities
        all_entities = [(name, 'consignor') for name in consignors] + [(name, 'buyer') for name in buyers]
        total_entities = len(all_entities)
        
        logger.info(f"Total entities to process: {total_entities}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Max consecutive timeouts: {self.max_consecutive_timeouts}")
        
        results = {}
        start_time = time.time()
        
        # Process in batches
        total_batches = (total_entities + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, total_entities, self.batch_size):
            batch_entities = all_entities[batch_idx:batch_idx + self.batch_size]
            batch_num = (batch_idx // self.batch_size) + 1
            
            logger.info(f"\n🚀 Processing batch {batch_num}/{total_batches}")
            
            for entity_name, entity_type in batch_entities:
                result = self.get_contact_info(entity_name, entity_type)
                
                # Check for exit condition
                if result is None:
                    logger.critical("Process terminated due to consecutive timeouts.")
                    break
                
                results[f"{entity_type}:{entity_name}"] = result
            
            # Exit if we hit timeout limit
            if self.consecutive_timeouts >= self.max_consecutive_timeouts:
                break
            
            # Delay between batches
            if batch_num < total_batches:
                logger.info(f"Waiting {self.batch_delay} seconds before next batch...")
                await asyncio.sleep(self.batch_delay)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Enrichment completed in {total_time/60:.1f} minutes")
        logger.info(f"📊 Final stats: {self.total_successful}/{total_entities} successful ({self.total_successful/total_entities*100:.1f}%)")
        
        return results

    # ==================== DATA INTEGRATION ====================
    
    def integrate_enrichment_data(self, df: pd.DataFrame, enrichment_results: Dict, 
                                 consignors: List[str], buyers: List[str]) -> pd.DataFrame:
        """Integrate enrichment results back into the main dataframe"""
        logger.info("🔗 Integrating enrichment data...")
        
        # Create lookup dictionaries for enrichment data
        consignor_enrichment = {}
        buyer_enrichment = {}
        
        for consignor in consignors:
            key = f"consignor:{consignor}"
            if key in enrichment_results:
                data = enrichment_results[key]
                consignor_enrichment[consignor] = {
                    'consignor_website': data.get('website'),
                    'consignor_email': data.get('email'),
                    'consignor_phone': data.get('phone'),
                    'consignor_city': data.get('city'),
                    'consignor_state': data.get('state'),
                    'consignor_country': data.get('country'),
                    'consignor_is_company': data.get('is_company')
                }
        
        for buyer in buyers:
            key = f"buyer:{buyer}"
            if key in enrichment_results:
                data = enrichment_results[key]
                buyer_enrichment[buyer] = {
                    'buyer_website': data.get('website'),
                    'buyer_email': data.get('email'),
                    'buyer_phone': data.get('phone'),
                    'buyer_city': data.get('city'),
                    'buyer_state': data.get('state'),
                    'buyer_country': data.get('country'),
                    'buyer_is_company': data.get('is_company')
                }
        
        # Update dataframe with enrichment data
        for idx, row in df.iterrows():
            consignor = row['consignor_cleaned']
            buyer = row['buyer_cleaned']
            
            if consignor in consignor_enrichment:
                for col, value in consignor_enrichment[consignor].items():
                    df.at[idx, col] = value
            
            if buyer in buyer_enrichment:
                for col, value in buyer_enrichment[buyer].items():
                    df.at[idx, col] = value
        
        # Calculate contact quality scores
        logger.info("📊 Calculating contact quality scores...")
        df['consignor_contact_score'] = df.apply(self.calculate_contact_score, axis=1, entity_type='consignor')
        df['buyer_contact_score'] = df.apply(self.calculate_contact_score, axis=1, entity_type='buyer')
        
        return df

    def calculate_contact_score(self, row, entity_type: str) -> int:
        """Calculate contact quality score (0-100) based on available information"""
        score = 0
        prefix = f"{entity_type}_"
        
        # Website: 30 points
        if pd.notna(row.get(f'{prefix}website')):
            score += 30
        
        # Email: 25 points
        if pd.notna(row.get(f'{prefix}email')):
            score += 25
        
        # Phone: 25 points
        if pd.notna(row.get(f'{prefix}phone')):
            score += 25
        
        # Location (city + state): 20 points
        if pd.notna(row.get(f'{prefix}city')) and pd.notna(row.get(f'{prefix}state')):
            score += 20
        elif pd.notna(row.get(f'{prefix}city')) or pd.notna(row.get(f'{prefix}state')):
            score += 10
        
        return score

    # ==================== RESULTS EXPORT ====================
    
    def save_results_to_csv(self, df: pd.DataFrame, enrichment_results: Dict) -> str:
        """Save final results to CSV format"""
        logger.info("💾 Saving results to CSV...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main dataset
        main_csv_filename = f'horse_sales_master_pipeline_{timestamp}.csv'
        df.to_csv(main_csv_filename, index=False)
        
        # Save enrichment summary CSV
        enrichment_csv_data = []
        
        for key, data in enrichment_results.items():
            entity_type, entity_name = key.split(':', 1)
            row = {
                'entity_name': entity_name,
                'entity_type': entity_type,
                'website': data.get('website'),
                'email': data.get('email'),
                'phone': data.get('phone'),
                'city': data.get('city'),
                'state': data.get('state'),
                'country': data.get('country'),
                'is_company': data.get('is_company')
            }
            enrichment_csv_data.append(row)
        
        enrichment_df = pd.DataFrame(enrichment_csv_data)
        enrichment_csv_filename = f'enrichment_summary_{timestamp}.csv'
        enrichment_df.to_csv(enrichment_csv_filename, index=False)
        
        # Save JSON results for reference
        json_filename = f'enrichment_results_{timestamp}.json'
        with open(json_filename, 'w') as f:
            json.dump(enrichment_results, f, indent=2)
        
        # Calculate and log statistics
        websites_found = enrichment_df['website'].notna().sum()
        emails_found = enrichment_df['email'].notna().sum()
        phones_found = enrichment_df['phone'].notna().sum()
        locations_found = enrichment_df['city'].notna().sum()
        
        logger.info(f"✅ Results saved:")
        logger.info(f"  📄 Main dataset: {main_csv_filename}")
        logger.info(f"  📄 Enrichment summary: {enrichment_csv_filename}")
        logger.info(f"  📄 JSON results: {json_filename}")
        logger.info(f"📊 Enrichment success rates:")
        logger.info(f"  Websites: {websites_found}/{len(enrichment_df)} ({websites_found/len(enrichment_df)*100:.1f}%)")
        logger.info(f"  Emails: {emails_found}/{len(enrichment_df)} ({emails_found/len(enrichment_df)*100:.1f}%)")
        logger.info(f"  Phones: {phones_found}/{len(enrichment_df)} ({phones_found/len(enrichment_df)*100:.1f}%)")
        logger.info(f"  Locations: {locations_found}/{len(enrichment_df)} ({locations_found/len(enrichment_df)*100:.1f}%)")
        
        return main_csv_filename

    # ==================== MASTER PIPELINE EXECUTION ====================
    
    async def run_master_pipeline(self) -> str:
        """Run the complete master pipeline"""
        logger.info("🎯 Starting Master Horse Sales Data Pipeline...")
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Query BigQuery
            logger.info("\n" + "="*80)
            logger.info("STEP 1: BIGQUERY DATA EXTRACTION")
            logger.info("="*80)
            raw_data = self.query_bigquery_data()
            
            # Step 2: Clean and process data
            logger.info("\n" + "="*80)
            logger.info("STEP 2: DATA CLEANING AND PROCESSING")
            logger.info("="*80)
            clean_data = self.clean_and_process_data(raw_data)
            
            # Step 3: Get all entities
            logger.info("\n" + "="*80)
            logger.info("STEP 3: GET ALL ENTITIES")
            logger.info("="*80)
            consignors, buyers = self.get_all_entities(clean_data)
            
            # Step 4: Web search enrichment
            logger.info("\n" + "="*80)
            logger.info("STEP 4: WEB SEARCH ENRICHMENT")
            logger.info("="*80)
            enrichment_results = await self.enrich_entities(consignors, buyers)
            
            # Step 5: Data integration
            logger.info("\n" + "="*80)
            logger.info("STEP 5: DATA INTEGRATION")
            logger.info("="*80)
            final_data = self.integrate_enrichment_data(clean_data, enrichment_results, consignors, buyers)
            
            # Step 6: Save results
            logger.info("\n" + "="*80)
            logger.info("STEP 6: RESULTS EXPORT")
            logger.info("="*80)
            output_filename = self.save_results_to_csv(final_data, enrichment_results)
            
            # Final summary
            total_time = time.time() - pipeline_start_time
            logger.info("\n" + "="*80)
            logger.info("🎉 MASTER PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
            logger.info(f"📊 Total records processed: {len(final_data):,}")
            logger.info(f"🎯 Entities enriched: {len(enrichment_results)}")
            logger.info(f"📄 Main output file: {output_filename}")
            
            return output_filename
            
        except Exception as e:
            logger.error(f"Master pipeline failed: {e}")
            raise

async def main():
    """Main function to run the master pipeline"""
    
    # Configuration
    gemini_api_key = "AIzaSyBB2-SGq0Pjbp4QDEBLVcejzz0LWg1O3_Q"
    bigquery_credentials_path = "/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/thorobid-dev-97dc54cf1d04.json"
    
    # Initialize and run master pipeline
    pipeline = MasterHorseSalesPipeline(gemini_api_key, bigquery_credentials_path)
    
    try:
        # Run with ALL entities
        output_filename = await pipeline.run_master_pipeline()
        print(f"\n🎉 SUCCESS! Master pipeline completed.")
        print(f"📄 Main results: {output_filename}")
        print(f"📋 Check log file: {log_filename}")
        
    except Exception as e:
        logger.error(f"Master pipeline execution failed: {e}")
        print(f"❌ Pipeline failed. Check log file: {log_filename}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())





    