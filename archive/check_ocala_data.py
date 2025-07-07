from google.cloud import bigquery
import os

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/thorobid-dev-97dc54cf1d04.json'

# Initialize client
client = bigquery.Client()

# Check Ocala data specifically
query = """
SELECT 
    consignor,
    buyer_name,
    sale_price,
    source_year,
    COUNT(*) as count
FROM `thorobid-dev.ingest.ocala_sales`
WHERE consignor IS NOT NULL
GROUP BY consignor, buyer_name, sale_price, source_year
ORDER BY count DESC
LIMIT 20
"""

try:
    print("Checking Ocala data structure...")
    results = client.query(query)
    df = results.to_dataframe()
    print("Top consignor/buyer combinations:")
    print(df.to_string(index=False))
    
    # Check for NULL values
    null_check = """
    SELECT 
        COUNT(*) as total_records,
        COUNT(consignor) as consignor_not_null,
        COUNT(buyer_name) as buyer_not_null,
        COUNT(sale_price) as price_not_null,
        COUNT(source_year) as year_not_null
    FROM `thorobid-dev.ingest.ocala_sales`
    """
    
    print("\nNull value check:")
    null_results = client.query(null_check)
    null_df = null_results.to_dataframe()
    print(null_df.to_string(index=False))
    
    # Sample of actual data
    sample_query = """
    SELECT 
        consignor,
        buyer_name,
        sale_price,
        source_year
    FROM `thorobid-dev.ingest.ocala_sales`
    WHERE consignor IS NOT NULL AND buyer_name IS NOT NULL
    LIMIT 10
    """
    
    print("\nSample Ocala records:")
    sample_results = client.query(sample_query)
    sample_df = sample_results.to_dataframe()
    print(sample_df.to_string(index=False))
    
except Exception as e:
    print(f"Error: {e}")