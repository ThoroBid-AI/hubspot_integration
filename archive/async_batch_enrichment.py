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

# Configure console logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup file logging
log_filename = f"enrichment_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.info(f"Process log file created: {log_filename}")

class AsyncBatchEnricher:
    def __init__(self, api_key: str, checkpoint_file: str = "enrichment_checkpoint.pkl"):
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
        
        # Batch settings
        self.batch_size = 3  # Process 3 entities per batch to stay under rate limits
        self.batch_delay = 120  # 2 minutes between batches
        self.call_interval = 8  # 8 seconds between individual calls
        
        # Resume capability
        self.checkpoint_file = checkpoint_file
        self.results = {}
        self.processed_entities = set()
        self.last_api_call = 0
        
        # Timeout tracking
        self.consecutive_timeouts = 0
        self.max_consecutive_timeouts = 5
        self.total_processed = 0
        self.total_successful = 0
        self.total_failed = 0
        
        # Load existing progress if available
        self.load_checkpoint()
        
        logger.info(f"AsyncBatchEnricher initialized - Max consecutive timeouts: {self.max_consecutive_timeouts}")
    
    def load_checkpoint(self):
        """Load previous progress from checkpoint file"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    self.results = checkpoint.get('results', {})
                    self.processed_entities = checkpoint.get('processed_entities', set())
                    logger.info(f"Loaded checkpoint: {len(self.processed_entities)} entities already processed")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                self.results = {}
                self.processed_entities = set()
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file"""
        try:
            checkpoint = {
                'results': self.results,
                'processed_entities': self.processed_entities,
                'timestamp': time.time()
            }
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            logger.info(f"Checkpoint saved: {len(self.processed_entities)} entities processed")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def rate_limit(self):
        """Ensure we don't exceed rate limits"""
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
        
        # Check if already processed
        entity_key = f"{entity_type}:{entity_name}"
        if entity_key in self.processed_entities:
            logger.info(f"Skipping already processed: {entity_name}")
            return self.results.get(entity_key, self.empty_result())
        
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
                
                # Store result and mark as processed
                self.results[entity_key] = result
                self.processed_entities.add(entity_key)
                
                # Reset consecutive timeouts on success
                self.consecutive_timeouts = 0
                self.total_processed += 1
                self.total_successful += 1
                
                logger.info(f"✓ Successfully processed: {entity_name} (Processed: {self.total_processed}, Success: {self.total_successful}, Failed: {self.total_failed})")
                return result
            else:
                logger.warning(f"No response received for {entity_name}")
                result = self.empty_result()
                self.results[entity_key] = result
                self.processed_entities.add(entity_key)
                self.total_processed += 1
                self.total_failed += 1
                return result
                
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
                    self.save_checkpoint()
                    sys.exit(1)
                
                logger.info(f"Timeout detected, sleeping for 60 seconds... ({self.consecutive_timeouts}/{self.max_consecutive_timeouts} consecutive)")
                time.sleep(60)
            else:
                # Reset consecutive timeouts for non-timeout errors
                self.consecutive_timeouts = 0
            
            result = self.empty_result()
            self.results[entity_key] = result
            self.processed_entities.add(entity_key)
            self.total_processed += 1
            self.total_failed += 1
            return result
    
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
    
    def process_batch(self, entities: List[tuple], batch_num: int, total_batches: int) -> Dict:
        """Process a batch of entities"""
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING BATCH {batch_num}/{total_batches}")
        logger.info(f"Entities: {[name for name, _ in entities]}")
        logger.info(f"{'='*60}")
        
        batch_results = {}
        
        for i, (entity_name, entity_type) in enumerate(entities, 1):
            logger.info(f"\n[{i}/{len(entities)}] Processing {entity_type}: {entity_name}")
            
            try:
                result = self.get_contact_info(entity_name, entity_type)
                batch_results[f"{entity_type}:{entity_name}"] = result
                
                # Save checkpoint after each entity
                self.save_checkpoint()
                
            except Exception as e:
                logger.error(f"Error processing {entity_name}: {e}")
                batch_results[f"{entity_type}:{entity_name}"] = self.empty_result()
        
        return batch_results
    
    async def process_all_entities(self, consignors: List[str], buyers: List[str]) -> Dict:
        """Process all entities in batches with delays"""
        
        # Combine all entities with their types
        all_entities = [(name, 'consignor') for name in consignors] + [(name, 'buyer') for name in buyers]
        
        # Filter out already processed entities
        remaining_entities = [
            (name, entity_type) for name, entity_type in all_entities 
            if f"{entity_type}:{name}" not in self.processed_entities
        ]
        
        total_entities = len(all_entities)
        remaining_count = len(remaining_entities)
        processed_count = total_entities - remaining_count
        
        logger.info(f"\n{'='*80}")
        logger.info(f"ASYNC BATCH ENRICHMENT - ENHANCED WITH TIMEOUT PROTECTION")
        logger.info(f"{'='*80}")
        logger.info(f"Total entities: {total_entities}")
        logger.info(f"Already processed: {processed_count}")
        logger.info(f"Remaining to process: {remaining_count}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Max consecutive timeouts before exit: {self.max_consecutive_timeouts}")
        logger.info(f"Estimated time: {remaining_count * self.call_interval / 60:.1f} minutes")
        
        if remaining_count == 0:
            logger.info("All entities already processed!")
            return self.results
        
        # Process in batches
        total_batches = (remaining_count + self.batch_size - 1) // self.batch_size
        start_time = time.time()
        
        for batch_idx in range(0, remaining_count, self.batch_size):
            batch_entities = remaining_entities[batch_idx:batch_idx + self.batch_size]
            batch_num = (batch_idx // self.batch_size) + 1
            
            batch_start_time = time.time()
            logger.info(f"\n🚀 Starting batch {batch_num}/{total_batches} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Process this batch
            batch_results = self.process_batch(batch_entities, batch_num, total_batches)
            
            # Update overall results
            self.results.update(batch_results)
            
            # Save checkpoint after each batch
            self.save_checkpoint()
            
            batch_time = time.time() - batch_start_time
            elapsed_total = time.time() - start_time
            
            logger.info(f"\n📊 Batch {batch_num} completed in {batch_time:.1f} seconds")
            logger.info(f"📈 Overall progress: {self.total_processed}/{total_entities} processed, {self.total_successful} successful, {self.total_failed} failed")
            logger.info(f"⏱️  Total elapsed time: {elapsed_total/60:.1f} minutes")
            
            # Check if we should exit due to timeouts (this check happens in get_contact_info)
            if self.consecutive_timeouts >= self.max_consecutive_timeouts:
                logger.critical("Process terminated due to consecutive timeouts.")
                return self.results
            
            # Delay between batches (except for last batch)
            if batch_num < total_batches:
                logger.info(f"\n⏱️  Batch {batch_num} complete. Waiting {self.batch_delay} seconds before next batch...")
                await asyncio.sleep(self.batch_delay)
        
        total_time = time.time() - start_time
        logger.info(f"\n✅ All entities processed successfully in {total_time/60:.1f} minutes!")
        logger.info(f"📊 Final stats: {self.total_successful}/{total_entities} successful ({self.total_successful/total_entities*100:.1f}%)")
        return self.results
    
    def generate_final_report(self, consignors: List[str], buyers: List[str]):
        """Generate comprehensive final report"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL ENRICHMENT REPORT")
        logger.info(f"{'='*80}")
        
        # Separate results by type
        consignor_results = {name: self.results.get(f"consignor:{name}", {}) for name in consignors}
        buyer_results = {name: self.results.get(f"buyer:{name}", {}) for name in buyers}
        
        # Calculate success rates
        def count_successes(results_dict):
            websites = sum(1 for r in results_dict.values() if r.get('website'))
            emails = sum(1 for r in results_dict.values() if r.get('email'))
            phones = sum(1 for r in results_dict.values() if r.get('phone'))
            locations = sum(1 for r in results_dict.values() if r.get('city'))
            return websites, emails, phones, locations
        
        c_websites, c_emails, c_phones, c_locations = count_successes(consignor_results)
        b_websites, b_emails, b_phones, b_locations = count_successes(buyer_results)
        
        print(f"\n📊 SUCCESS RATES:")
        print(f"\nConsignors (15 entities):")
        print(f"  Websites: {c_websites}/15 ({c_websites/15*100:.1f}%)")
        print(f"  Emails: {c_emails}/15 ({c_emails/15*100:.1f}%)")
        print(f"  Phones: {c_phones}/15 ({c_phones/15*100:.1f}%)")
        print(f"  Locations: {c_locations}/15 ({c_locations/15*100:.1f}%)")
        
        print(f"\nBuyers (15 entities):")
        print(f"  Websites: {b_websites}/15 ({b_websites/15*100:.1f}%)")
        print(f"  Emails: {b_emails}/15 ({b_emails/15*100:.1f}%)")
        print(f"  Phones: {b_phones}/15 ({b_phones/15*100:.1f}%)")
        print(f"  Locations: {b_locations}/15 ({b_locations/15*100:.1f}%)")
        
        # Show successful examples
        print(f"\n🎯 SUCCESSFUL ENRICHMENTS:")
        
        print(f"\nConsignors with complete contact info:")
        for name, info in consignor_results.items():
            if info.get('website') or info.get('email') or info.get('phone'):
                print(f"\n  ✅ {name}:")
                for key, value in info.items():
                    if value and key != 'is_company':
                        print(f"     {key}: {value}")
        
        print(f"\nBuyers with complete contact info:")
        for name, info in buyer_results.items():
            if info.get('website') or info.get('email') or info.get('phone'):
                print(f"\n  ✅ {name}:")
                for key, value in info.items():
                    if value and key != 'is_company':
                        print(f"     {key}: {value}")
        
        # Save detailed results
        results_file = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/async_enrichment_results.json'
        final_results = {
            'consignors': consignor_results,
            'buyers': buyer_results,
            'summary': {
                'total_entities': 30,
                'consignor_success_rates': {
                    'websites': f"{c_websites}/15 ({c_websites/15*100:.1f}%)",
                    'emails': f"{c_emails}/15 ({c_emails/15*100:.1f}%)",
                    'phones': f"{c_phones}/15 ({c_phones/15*100:.1f}%)",
                    'locations': f"{c_locations}/15 ({c_locations/15*100:.1f}%)"
                },
                'buyer_success_rates': {
                    'websites': f"{b_websites}/15 ({b_websites/15*100:.1f}%)",
                    'emails': f"{b_emails}/15 ({b_emails/15*100:.1f}%)",
                    'phones': f"{b_phones}/15 ({b_phones/15*100:.1f}%)",
                    'locations': f"{b_locations}/15 ({b_locations/15*100:.1f}%)"
                }
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: async_enrichment_results.json")
        print(f"\n🏁 ENRICHMENT COMPLETE!")

def clean_csv_data():
    """Clean the CSV data before processing"""
    logger.info("🧹 Starting CSV data cleaning...")
    
    try:
        # Load the dataset
        df = pd.read_csv('/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_enriched.csv')
        original_count = len(df)
        logger.info(f"Loaded {original_count} records from CSV")
        
        # Remove duplicates
        df_clean = df.drop_duplicates()
        after_dedup = len(df_clean)
        logger.info(f"Removed {original_count - after_dedup} duplicate records")
        
        # Clean consignor and buyer names
        df_clean['consignor_cleaned'] = df_clean['consignor_cleaned'].fillna('').str.strip()
        df_clean['buyer_cleaned'] = df_clean['buyer_cleaned'].fillna('').str.strip()
        
        # Remove empty consignor/buyer entries
        df_clean = df_clean[
            (df_clean['consignor_cleaned'] != '') & 
            (df_clean['buyer_cleaned'] != '')
        ]
        after_cleaning = len(df_clean)
        logger.info(f"Removed {after_dedup - after_cleaning} records with empty consignor/buyer names")
        
        # Save cleaned data back
        backup_file = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_enriched_backup.csv'
        df.to_csv(backup_file, index=False)
        logger.info(f"Original data backed up to: {backup_file}")
        
        df_clean.to_csv('/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_enriched.csv', index=False)
        logger.info(f"✅ CSV cleaning complete: {original_count} → {after_cleaning} records ({(after_cleaning/original_count)*100:.1f}% retained)")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error during CSV cleaning: {e}")
        # If cleaning fails, load original data
        return pd.read_csv('/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_enriched.csv')

async def main():
    """Main async function to run the enrichment"""
    
    # Clean CSV data first
    df = clean_csv_data()
    
    # Load dataset and select top entities
    logger.info("📊 Selecting top entities for enrichment...")
    
    # Get entities from past 3 years for more relevance
    recent_data = df[df['sale_year'].isin([2023, 2024, 2025])]
    
    # Select top 15 consignors by frequency
    top_consignors = recent_data['consignor_cleaned'].value_counts().head(15).index.tolist()
    
    # Select top 15 buyers by frequency (excluding non-buyers)
    actual_buyers = recent_data[
        recent_data['buyer_cleaned'].notna() & 
        ~recent_data['buyer_cleaned'].isin(['NOT SOLD', 'OUT', 'WITHDRAWN', '', 'RNA'])
    ]
    top_buyers = actual_buyers['buyer_cleaned'].value_counts().head(15).index.tolist()
    
    logger.info(f"Selected entities:")
    logger.info(f"Consignors: {top_consignors}")
    logger.info(f"Buyers: {top_buyers}")
    
    # Initialize enricher
    api_key = "AIzaSyBB2-SGq0Pjbp4QDEBLVcejzz0LWg1O3_Q"
    enricher = AsyncBatchEnricher(api_key)
    
    # Process all entities
    await enricher.process_all_entities(top_consignors, top_buyers)
    
    # Generate final report
    enricher.generate_final_report(top_consignors, top_buyers)

if __name__ == "__main__":
    asyncio.run(main())