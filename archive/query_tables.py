from google.cloud import bigquery
import os

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/thorobid-dev-97dc54cf1d04.json'

# Initialize client
client = bigquery.Client()

# Define the three tables
tables = [
    'thorobid-dev.ingest.fasigtipton_sales',
    'thorobid-dev.ingest.keeneland_sales',
    'thorobid-dev.ingest.ocala_sales'
]

# Query each table
for table in tables:
    print(f"\n{'='*50}")
    print(f"Querying {table}")
    print(f"{'='*50}")
    
    query = f"SELECT * FROM `{table}` LIMIT 10"
    
    try:
        results = client.query(query)
        df = results.to_dataframe()
        print(f"Table has {len(df)} rows (showing first 10)")
        print(f"Columns: {list(df.columns)}")
        print("\nSample data:")
        print(df.to_string())
    except Exception as e:
        print(f"Error querying {table}: {e}")