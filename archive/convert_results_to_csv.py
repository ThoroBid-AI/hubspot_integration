import json
import pandas as pd

def convert_enrichment_results_to_csv():
    """Convert the JSON enrichment results to CSV format"""
    
    # Load the JSON results
    with open('async_enrichment_results.json', 'r') as f:
        data = json.load(f)
    
    # Prepare data for CSV
    csv_data = []
    
    # Process consignors
    for name, info in data['consignors'].items():
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
    for name, info in data['buyers'].items():
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
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv('enrichment_results.csv', index=False)
    
    print(f"✅ Converted {len(csv_data)} entities to CSV")
    print("📄 File saved as: enrichment_results.csv")
    
    # Show preview
    print("\n📊 Preview of results:")
    print(df.head(10).to_string(index=False))
    
    # Show success statistics
    websites_found = df['website'].notna().sum()
    emails_found = df['email'].notna().sum()
    phones_found = df['phone'].notna().sum()
    
    print(f"\n📈 Success rates:")
    print(f"Websites: {websites_found}/{len(df)} ({websites_found/len(df)*100:.1f}%)")
    print(f"Emails: {emails_found}/{len(df)} ({emails_found/len(df)*100:.1f}%)")
    print(f"Phones: {phones_found}/{len(df)} ({phones_found/len(df)*100:.1f}%)")

if __name__ == "__main__":
    convert_enrichment_results_to_csv()