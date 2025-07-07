#!/usr/bin/env python3
"""
Complete Horse Sales Data Pipeline - CSV Version
===============================================
1. Load existing CSV data
2. Clean and process the data  
3. Select top 30 buyers and consignors
4. Run web search enrichment with timeout protection
5. Generate final CSV results

Usage: python complete_pipeline_csv.py
"""

import pandas as pd
import google.generativeai as genai
import asyncio
import os
import time
import re
import json
from typing import Dict, Optional, List
import logging
import pickle
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup file logging
log_filename = f"complete_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.info(f"Complete pipeline log file created: {log_filename}")

class HorseSalesDataPipeline:
    def __init__(self, gemini_api_key: str, csv_file_path: str):
        self.gemini_api_key = gemini_api_key
        self.csv_file_path = csv_file_path
        
        # Configure Gemini
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
        
        # Enrichment settings
        self.batch_size = 3
        self.batch_delay = 120
        self.call_interval = 8
        self.max_consecutive_timeouts = 5
        
        # Tracking variables
        self.consecutive_timeouts = 0
        self.total_processed = 0
        self.total_successful = 0
        self.total_failed = 0
        self.results = {}
        self.last_api_call = 0
        
        logger.info("Pipeline initialized successfully")

    def load_csv_data(self) -> pd.DataFrame:
        """Load CSV data from file"""
        logger.info(f"📁 Loading CSV data from: {self.csv_file_path}")
        
        try:
            df = pd.read_csv(self.csv_file_path)
            logger.info(f"✅ Loaded {len(df)} records from CSV")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def clean_and_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process the horse sales data"""
        logger.info("🧹 Cleaning and processing data...")
        
        original_count = len(df)
        
        # Create backup
        backup_file = self.csv_file_path.replace('.csv', '_backup.csv')
        df.to_csv(backup_file, index=False)
        logger.info(f"📋 Backup created: {backup_file}")
        
        # Remove duplicates
        df_clean = df.drop_duplicates()
        after_dedup = len(df_clean)
        logger.info(f"Removed {original_count - after_dedup} duplicate records")
        
        # Clean consignor and buyer names
        if 'consignor_cleaned' in df_clean.columns:
            df_clean['consignor_cleaned'] = df_clean['consignor_cleaned'].fillna('').str.strip()
        if 'buyer_cleaned' in df_clean.columns:
            df_clean['buyer_cleaned'] = df_clean['buyer_cleaned'].fillna('').str.strip()
        
        # Remove common non-buyer values
        non_buyers = ['NOT SOLD', 'OUT', 'WITHDRAWN', 'RNA', 'PASSED', 'BUY BACK', '']
        if 'buyer_cleaned' in df_clean.columns:
            df_clean = df_clean[~df_clean['buyer_cleaned'].isin(non_buyers)]
        
        # Remove empty entries
        if 'consignor_cleaned' in df_clean.columns and 'buyer_cleaned' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['consignor_cleaned'] != '') & 
                (df_clean['buyer_cleaned'] != '') &
                (df_clean['consignor_cleaned'].notna()) &
                (df_clean['buyer_cleaned'].notna())
            ]
        
        after_cleaning = len(df_clean)
        logger.info(f"After cleaning: {original_count} → {after_cleaning} records ({(after_cleaning/original_count)*100:.1f}% retained)")
        
        return df_clean

    def select_top_entities(self, df: pd.DataFrame) -> tuple:
        """Select top 15 consignors and 15 buyers from recent years"""
        logger.info("📊 Selecting top entities...")
        
        # Focus on recent years if available
        if 'sale_year' in df.columns:
            recent_data = df[df['sale_year'].isin([2023, 2024, 2025])]
            if len(recent_data) == 0:
                recent_data = df  # Use all data if no recent data
        else:
            recent_data = df
        
        logger.info(f"Using {len(recent_data)} records for entity selection")
        
        # Get top 15 consignors by frequency
        if 'consignor_cleaned' in recent_data.columns:
            top_consignors = recent_data['consignor_cleaned'].value_counts().head(15).index.tolist()
        else:
            top_consignors = []
        
        # Get top 15 buyers by frequency
        if 'buyer_cleaned' in recent_data.columns:
            top_buyers = recent_data['buyer_cleaned'].value_counts().head(15).index.tolist()
        else:
            top_buyers = []
        
        logger.info(f"Selected {len(top_consignors)} consignors and {len(top_buyers)} buyers")
        logger.info(f"Top consignors: {top_consignors[:5]}...")
        logger.info(f"Top buyers: {top_buyers[:5]}...")
        
        return top_consignors, top_buyers

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
        """Get contact information for a single entity"""
        if not entity_name or entity_name.strip() == "":
            return self.empty_result()

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
                result = self.parse_response(response.text, entity_name)
                
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
                return self.empty_result()

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
                    sys.exit(1)
                
                logger.info(f"Timeout detected, sleeping for 60 seconds... ({self.consecutive_timeouts}/{self.max_consecutive_timeouts} consecutive)")
                time.sleep(60)
            else:
                # Reset consecutive timeouts for non-timeout errors
                self.consecutive_timeouts = 0
            
            self.total_processed += 1
            self.total_failed += 1
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
                
                # Log successful findings
                found_items = [k for k, v in result.items() if v]
                if found_items:
                    logger.info(f"  Found: {', '.join(found_items)}")
                
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
                results[f"{entity_type}:{entity_name}"] = result
                
                # Check for exit condition
                if self.consecutive_timeouts >= self.max_consecutive_timeouts:
                    logger.critical("Process terminated due to consecutive timeouts.")
                    break
            
            # Delay between batches
            if batch_num < total_batches and self.consecutive_timeouts < self.max_consecutive_timeouts:
                logger.info(f"Waiting {self.batch_delay} seconds before next batch...")
                await asyncio.sleep(self.batch_delay)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Enrichment completed in {total_time/60:.1f} minutes")
        logger.info(f"📊 Final stats: {self.total_successful}/{total_entities} successful ({self.total_successful/total_entities*100:.1f}%)")
        
        return results

    def save_results_to_csv(self, results: Dict, consignors: List[str], buyers: List[str]) -> str:
        """Save enrichment results to CSV format"""
        logger.info("💾 Saving results to CSV...")
        
        csv_data = []
        
        # Process consignors
        for name in consignors:
            key = f"consignor:{name}"
            info = results.get(key, self.empty_result())
            row = {
                'entity_name': name,
                'entity_type': 'consignor',
                'website': info.get('website'),
                'email': info.get('email'),
                'phone': info.get('phone'),
                'city': info.get('city'),
                'state': info.get('state'),
                'country': info.get('country'),
                'is_company': info.get('is_company')
            }
            csv_data.append(row)
        
        # Process buyers
        for name in buyers:
            key = f"buyer:{name}"
            info = results.get(key, self.empty_result())
            row = {
                'entity_name': name,
                'entity_type': 'buyer',
                'website': info.get('website'),
                'email': info.get('email'),
                'phone': info.get('phone'),
                'city': info.get('city'),
                'state': info.get('state'),
                'country': info.get('country'),
                'is_company': info.get('is_company')
            }
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'horse_sales_enriched_results_{timestamp}.csv'
        df.to_csv(csv_filename, index=False)
        
        # Calculate statistics
        websites_found = df['website'].notna().sum()
        emails_found = df['email'].notna().sum()
        phones_found = df['phone'].notna().sum()
        locations_found = df['city'].notna().sum()
        
        logger.info(f"✅ Results saved to: {csv_filename}")
        logger.info(f"📊 Success rates:")
        logger.info(f"  Websites: {websites_found}/{len(df)} ({websites_found/len(df)*100:.1f}%)")
        logger.info(f"  Emails: {emails_found}/{len(df)} ({emails_found/len(df)*100:.1f}%)")
        logger.info(f"  Phones: {phones_found}/{len(df)} ({phones_found/len(df)*100:.1f}%)")
        logger.info(f"  Locations: {locations_found}/{len(df)} ({locations_found/len(df)*100:.1f}%)")
        
        # Show preview of successful enrichments
        successful_entities = df[(df['website'].notna()) | (df['email'].notna()) | (df['phone'].notna())]
        logger.info(f"\n🎯 Successfully enriched {len(successful_entities)} entities:")
        for _, row in successful_entities.head(10).iterrows():
            contact_info = []
            if row['website']: contact_info.append(f"website: {row['website']}")
            if row['email']: contact_info.append(f"email: {row['email']}")
            if row['phone']: contact_info.append(f"phone: {row['phone']}")
            if contact_info:
                logger.info(f"  ✅ {row['entity_name']} ({row['entity_type']}): {', '.join(contact_info)}")
        
        return csv_filename

    async def run_complete_pipeline(self) -> str:
        """Run the complete pipeline from CSV to enriched CSV results"""
        logger.info("🎯 Starting complete horse sales data pipeline...")
        
        try:
            # Step 1: Load CSV data
            logger.info("\n" + "="*80)
            logger.info("STEP 1: LOADING CSV DATA")
            logger.info("="*80)
            raw_data = self.load_csv_data()
            
            # Step 2: Clean and process data
            logger.info("\n" + "="*80)
            logger.info("STEP 2: CLEANING DATA")
            logger.info("="*80)
            clean_data = self.clean_and_process_data(raw_data)
            
            # Step 3: Select top entities
            logger.info("\n" + "="*80)
            logger.info("STEP 3: SELECTING TOP ENTITIES")
            logger.info("="*80)
            consignors, buyers = self.select_top_entities(clean_data)
            
            # Step 4: Enrich with contact information
            logger.info("\n" + "="*80)
            logger.info("STEP 4: WEB SEARCH ENRICHMENT")
            logger.info("="*80)
            enrichment_results = await self.enrich_entities(consignors, buyers)
            
            # Step 5: Save to CSV
            logger.info("\n" + "="*80)
            logger.info("STEP 5: SAVING RESULTS")
            logger.info("="*80)
            csv_filename = self.save_results_to_csv(enrichment_results, consignors, buyers)
            
            logger.info("\n" + "="*80)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"📄 Final results: {csv_filename}")
            
            return csv_filename
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

async def main():
    """Main function to run the complete pipeline"""
    
    # Configuration
    gemini_api_key = "AIzaSyBB2-SGq0Pjbp4QDEBLVcejzz0LWg1O3_Q"
    csv_file_path = "/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_enriched.csv"
    
    # Initialize and run pipeline
    pipeline = HorseSalesDataPipeline(gemini_api_key, csv_file_path)
    
    try:
        csv_filename = await pipeline.run_complete_pipeline()
        print(f"\n🎉 SUCCESS! Results saved to: {csv_filename}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"❌ Pipeline failed. Check log file: {log_filename}")

if __name__ == "__main__":
    asyncio.run(main())