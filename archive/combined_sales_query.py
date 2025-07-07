from google.cloud import bigquery
import os

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/thorobid-dev-97dc54cf1d04.json'

# Initialize client
client = bigquery.Client()

# Combined query with all three tables
query = """
CREATE OR REPLACE TABLE `thorobid-dev.ingest.saleshubspot_combined_to_delete` AS

SELECT
  REPLACE(consignor_central, ',', '') as consignor,
  REPLACE(buyer, ',', '') as buyer,
  sale_price,
  CAST(SPLIT(sale, ' ')[OFFSET(1)] AS INT64) as sale_year,
  'keeneland' as source
FROM `thorobid-dev.ingest.keeneland_sales`

UNION ALL

SELECT
  REPLACE(consignor_name, ',', '') as consignor,
  REPLACE(purchaser, ',', '') as buyer,
  price as sale_price,
  EXTRACT(YEAR FROM session) as sale_year,
  'fasig' as source
FROM `thorobid-dev.ingest.fasigtipton_sales`

UNION ALL

SELECT
  REPLACE(consignor, ',', '') as consignor,
  REPLACE(buyer_name, ',', '') as buyer,
  sale_price,
  CAST(source_year AS INT64) as sale_year,
  'ocala' as source
FROM `thorobid-dev.ingest.ocala_sales`
"""

try:
    print("Running combined sales query...")
    job = client.query(query)
    job.result()  # Wait for query to complete
    print("✓ Table created successfully!")
    
    # Query the results to verify
    verify_query = """
    SELECT 
        source,
        COUNT(*) as record_count,
        COUNT(DISTINCT consignor) as unique_consignors,
        COUNT(DISTINCT buyer) as unique_buyers,
        MIN(sale_year) as min_year,
        MAX(sale_year) as max_year
    FROM `thorobid-dev.ingest.saleshubspot_combined_to_delete`
    GROUP BY source
    ORDER BY source
    """
    
    print("\nVerifying results:")
    results = client.query(verify_query)
    df = results.to_dataframe()
    print(df.to_string(index=False))
    
    # Sample records from each source
    sample_query = """
    SELECT *
    FROM `thorobid-dev.ingest.saleshubspot_combined_to_delete`
    WHERE source = 'ocala'
    LIMIT 5
    """
    
    print("\nSample Ocala records:")
    sample_results = client.query(sample_query)
    sample_df = sample_results.to_dataframe()
    print(sample_df.to_string(index=False))
    
except Exception as e:
    print(f"Error: {e}")