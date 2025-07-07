#!/usr/bin/env python3
"""
Quick CSV generation for review - BigQuery union + basic cleaning only
"""

import pandas as pd
from google.cloud import bigquery
import os
import re
from datetime import datetime

# Initialize BigQuery client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/thorobid-dev-97dc54cf1d04.json'
bq_client = bigquery.Client()

print("🔍 Querying BigQuery for horse sales data...")

# Simplified query - get sample data from each table
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
  foaling_year,
  source
FROM (
  -- Ocala Sales (sample)
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
    CAST(foaling_year AS STRING) as foaling_year,
    'ocala' as source
  FROM `thorobid-dev.ingest.ocala_sales`
  LIMIT 1000

  UNION ALL

  -- Keeneland Sales (sample)
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
    CAST(yob AS STRING) as foaling_year,
    'keeneland' as source
  FROM `thorobid-dev.ingest.keeneland_sales`
  LIMIT 1000

  UNION ALL

  -- Fasig-Tipton Sales (sample)
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
    CAST(YEAR_OF_BIRTH AS STRING) as foaling_year,
    'fasigtipton' as source
  FROM `thorobid-dev.ingest.fasigtipton_sales`
  LIMIT 1000
)
ORDER BY foaling_year DESC
"""

print("Executing BigQuery union query (sample data)...")
df = bq_client.query(query).to_dataframe()
print(f"✅ Retrieved {len(df)} records from BigQuery")
print(f"Data sources: {df['source'].value_counts().to_dict()}")

def clean_agent_text(text):
    """Remove agent-related text from names"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    agent_patterns = [
        r'\b(agent|agt|as agent|as agt)\b',
        r'\(agent.*?\)',
        r'\[agent.*?\]',
        r'\bagent for\b',
        r'\bacting as agent\b'
    ]
    
    for pattern in agent_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up extra spaces and punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[,\(\)\[\]]+$', '', text)
    return text.strip()

print("🧹 Applying basic cleaning...")

# Basic cleaning - remove commas and clean agent text
df['consignor_cleaned'] = df['consignor'].fillna('').apply(clean_agent_text).str.replace(',', '', regex=False)
df['buyer_cleaned'] = df['buyer_name'].fillna('').apply(clean_agent_text).str.replace(',', '', regex=False)

# Remove empty entries
df = df[
    (df['consignor_cleaned'] != '') & 
    (df['buyer_cleaned'] != '') &
    (df['consignor_cleaned'].notna()) &
    (df['buyer_cleaned'].notna())
]

print(f"After cleaning: {len(df)} records")
print(f"Unique consignors: {df['consignor_cleaned'].nunique()}")
print(f"Unique buyers: {df['buyer_cleaned'].nunique()}")

# Save to CSV
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f'horse_sales_review_{timestamp}.csv'
df.to_csv(csv_filename, index=False)

print(f"💾 CSV file saved: {csv_filename}")
print(f"📊 Columns: {list(df.columns)}")

# Show sample
print(f"\n📄 Sample cleaned data:")
sample_data = df[['consignor', 'consignor_cleaned', 'buyer_name', 'buyer_cleaned', 'source']].head(10)
print(sample_data.to_string(index=False))

print(f"\n🎯 Review file ready: {csv_filename}")