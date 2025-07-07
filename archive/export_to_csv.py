from google.cloud import bigquery
import os

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/thorobid-dev-97dc54cf1d04.json'

# Initialize client
client = bigquery.Client()

# Query to export only the 5 specific columns
query = """
SELECT 
    consignor,
    buyer,
    sale_price,
    sale_year,
    source
FROM `thorobid-dev.ingest.saleshubspot_combined_to_delete`
ORDER BY source, sale_year, consignor
"""

try:
    print("Exporting combined sales data to CSV...")
    results = client.query(query)
    df = results.to_dataframe()
    
    # Clean the data before export
    df = df.fillna('')  # Replace NaN with empty string
    
    # Save to CSV with proper formatting
    csv_filename = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_v4.csv'
    df.to_csv(csv_filename, index=False, quoting=0)  # No quotes unless necessary
    
    print(f"✓ Data exported to: {csv_filename}")
    print(f"Total records: {len(df):,}")
    print("\nData summary:")
    print(df.groupby('source').size().to_string())
    
except Exception as e:
    print(f"Error: {e}")