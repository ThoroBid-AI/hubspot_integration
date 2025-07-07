from google.cloud import bigquery
import os

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/thorobid-dev-97dc54cf1d04.json'

# Initialize client
client = bigquery.Client()

# Check distinct source values
query = """
SELECT 
    source,
    COUNT(*) as count
FROM `thorobid-dev.ingest.saleshubspot_combined_to_delete`
GROUP BY source
ORDER BY source
"""

try:
    print("Checking distinct source values...")
    results = client.query(query)
    df = results.to_dataframe()
    print(df.to_string(index=False))
    
except Exception as e:
    print(f"Error: {e}")