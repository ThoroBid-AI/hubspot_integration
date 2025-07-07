from google.cloud import bigquery
import os
import pandas as pd

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/thorobid-dev-97dc54cf1d04.json'

# Initialize client
client = bigquery.Client()

# Query to debug specific records
query = """
SELECT 
    consignor,
    buyer,
    sale_price,
    sale_year,
    source
FROM `thorobid-dev.ingest.saleshubspot_combined_to_delete`
WHERE source = 'ocala'
LIMIT 10
"""

try:
    print("Debugging CSV alignment issues...")
    results = client.query(query)
    df = results.to_dataframe()
    
    print("Sample Ocala records:")
    print(df.to_string(index=False))
    
    # Check for any embedded commas or newlines that might break CSV
    print("\nChecking for problematic characters...")
    for col in df.columns:
        has_comma = df[col].astype(str).str.contains(',', na=False).any()
        has_newline = df[col].astype(str).str.contains('\n', na=False).any()
        has_quote = df[col].astype(str).str.contains('"', na=False).any()
        print(f"{col}: comma={has_comma}, newline={has_newline}, quote={has_quote}")
    
    # Try a more controlled export
    print("\nExporting with proper escaping...")
    csv_filename = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/test_export.csv'
    df.to_csv(csv_filename, index=False, quoting=1)  # Quote all fields
    print(f"Test export saved to: {csv_filename}")
    
except Exception as e:
    print(f"Error: {e}")