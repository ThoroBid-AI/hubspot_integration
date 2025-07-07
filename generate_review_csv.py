#!/usr/bin/env python3
"""
Generate CSV file for review by running BigQuery union and cleaning logic
WITHOUT any changes to the existing logic - just data extraction and cleaning
"""

import pandas as pd
from google.cloud import bigquery
import os
import re
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime

# Initialize BigQuery client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/thorobid-dev-97dc54cf1d04.json'
bq_client = bigquery.Client()

print("🔍 Querying BigQuery for horse sales data...")

# Exact query from master_pipeline.py
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
  source_year,
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
    CAST(source_year AS STRING) as source_year,
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
    NULL as source_year,
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
    NULL as source_year,
    'fasigtipton' as source
  FROM `thorobid-dev.ingest.fasigtipton_sales`
)
ORDER BY foaling_year DESC
"""

print("Executing BigQuery union query across all 3 tables...")
df = bq_client.query(query).to_dataframe()
print(f"✅ Retrieved {len(df)} records from BigQuery")
print(f"Data sources: {df['source'].value_counts().to_dict()}")

# Apply the exact cleaning logic from master_pipeline.py

# Data cleaning configuration (from master_pipeline.py)
similarity_threshold = 0.8
agent_patterns = [
    r'\b(agent|agt|as agent|as agt)\b',
    r'\(agent.*?\)',
    r'\[agent.*?\]',
    r'\bagent for\b',
    r'\bacting as agent\b'
]

business_suffixes = [
    'llc', 'inc', 'corp', 'ltd', 'farm', 'farms', 'stable', 'stables', 
    'racing', 'bloodstock', 'sales', 'training', 'stud', 'ranch',
    'partnership', 'syndicate', 'group', 'company', 'co', 'enterprises'
]

def clean_agent_text(text):
    """Remove agent-related text from names"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    for pattern in agent_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up extra spaces and punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[,\(\)\[\]]+$', '', text)
    return text.strip()

def normalize_name(name):
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

def similarity_score(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a, b).ratio()

def deduplicate_name(name, existing_names):
    """Find the best match for a name among existing names"""
    if not name:
        return name
    
    normalized = normalize_name(name)
    
    # Check for exact match first
    if normalized in existing_names:
        return existing_names[normalized]
    
    # Check for similar matches
    best_match = None
    best_score = 0
    
    for existing_norm, existing_orig in existing_names.items():
        score = similarity_score(normalized, existing_norm)
        if score > best_score and score >= similarity_threshold:
            best_score = score
            best_match = existing_orig
    
    if best_match:
        return best_match
    else:
        # Add new name to registry
        existing_names[normalized] = name
        return name

def determine_entity_type(name):
    """Determine if entity is a company (True) or individual (False)"""
    if not name or pd.isna(name):
        return None
    
    name_lower = str(name).lower()
    
    # Check for business suffixes
    for suffix in business_suffixes:
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

print("🧹 Starting comprehensive data cleaning...")

original_count = len(df)

# Remove duplicates
df_clean = df.drop_duplicates()
after_dedup = len(df_clean)
print(f"Removed {original_count - after_dedup} duplicate records")

# Clean agent text from names and remove commas
df_clean['consignor_cleaned'] = df_clean['consignor'].apply(clean_agent_text).str.replace(',', '', regex=False)
df_clean['buyer_cleaned'] = df_clean['buyer_name'].apply(clean_agent_text).str.replace(',', '', regex=False)

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
print(f"After basic cleaning: {original_count} → {after_cleaning} records")

# Deduplicate names
print("🔄 Deduplicating entity names...")
consignor_registry = {}
buyer_registry = {}

df_clean['consignor_cleaned'] = df_clean['consignor_cleaned'].apply(
    lambda x: deduplicate_name(x, consignor_registry)
)
df_clean['buyer_cleaned'] = df_clean['buyer_cleaned'].apply(
    lambda x: deduplicate_name(x, buyer_registry)
)

# Determine entity types
print("🏢 Determining entity types...")
df_clean['consignor_is_company'] = df_clean['consignor_cleaned'].apply(determine_entity_type)
df_clean['buyer_is_company'] = df_clean['buyer_cleaned'].apply(determine_entity_type)

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
print(f"✅ Data cleaning complete: {original_count} → {final_count} records ({(final_count/original_count)*100:.1f}% retained)")
print(f"Unique consignors: {df_clean['consignor_cleaned'].nunique()}")
print(f"Unique buyers: {df_clean['buyer_cleaned'].nunique()}")

# Save to CSV
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f'horse_sales_union_cleaned_{timestamp}.csv'
df_clean.to_csv(csv_filename, index=False)

print(f"💾 CSV file saved: {csv_filename}")
print(f"📊 Total columns: {len(df_clean.columns)}")
print(f"📋 Column list: {list(df_clean.columns)}")

# Show sample data
print(f"\n📄 Sample data (first 5 rows):")
print(df_clean.head().to_string())

print(f"\n🎯 CSV file ready for review: {csv_filename}")