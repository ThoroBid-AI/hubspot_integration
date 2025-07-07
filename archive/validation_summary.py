import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_validation_results(csv_file: str):
    """Analyze the website validation results"""
    logger.info(f"Loading validation results from {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Website validation summary
    total_consignor_websites = df['consignor_website'].notna().sum()
    valid_consignor_websites = df['consignor_website_valid'].sum()
    total_buyer_websites = df['buyer_website'].notna().sum()
    valid_buyer_websites = df['buyer_website_valid'].sum()
    
    print("\n" + "="*60)
    print("WEBSITE VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\nConsignor Websites:")
    print(f"  Total found: {total_consignor_websites:,}")
    print(f"  Valid (200 OK): {valid_consignor_websites:,}")
    print(f"  Invalid: {total_consignor_websites - valid_consignor_websites:,}")
    print(f"  Success rate: {valid_consignor_websites/total_consignor_websites*100:.1f}%" if total_consignor_websites > 0 else "  No websites found")
    
    print(f"\nBuyer Websites:")
    print(f"  Total found: {total_buyer_websites:,}")
    print(f"  Valid (200 OK): {valid_buyer_websites:,}")
    print(f"  Invalid: {total_buyer_websites - valid_buyer_websites:,}")
    print(f"  Success rate: {valid_buyer_websites/total_buyer_websites*100:.1f}%" if total_buyer_websites > 0 else "  No websites found")
    
    # Show valid websites examples
    valid_consignor_examples = df[
        (df['consignor_website'].notna()) & 
        (df['consignor_website_valid'] == True)
    ][['consignor_cleaned', 'consignor_website']].drop_duplicates().head(10)
    
    if not valid_consignor_examples.empty:
        print(f"\n" + "-"*60)
        print("VALID CONSIGNOR WEBSITES (Sample)")
        print("-"*60)
        for _, row in valid_consignor_examples.iterrows():
            print(f"  ✓ {row['consignor_cleaned']}: {row['consignor_website']}")
    
    # Show invalid websites examples
    invalid_consignor_examples = df[
        (df['consignor_website'].notna()) & 
        (df['consignor_website_valid'] == False)
    ][['consignor_cleaned', 'consignor_website', 'consignor_website_error']].drop_duplicates().head(10)
    
    if not invalid_consignor_examples.empty:
        print(f"\n" + "-"*60)
        print("INVALID CONSIGNOR WEBSITES (Sample)")
        print("-"*60)
        for _, row in invalid_consignor_examples.iterrows():
            error = row['consignor_website_error'] if pd.notna(row['consignor_website_error']) else 'Unknown error'
            print(f"  ✗ {row['consignor_cleaned']}: {row['consignor_website']} ({error})")
    
    # Status code breakdown
    status_counts = df['consignor_website_status'].value_counts().sort_index()
    if not status_counts.empty:
        print(f"\n" + "-"*60)
        print("HTTP STATUS CODE BREAKDOWN")
        print("-"*60)
        for status, count in status_counts.items():
            if pd.notna(status):
                print(f"  {int(status)}: {count:,} websites")
    
    # Error type breakdown
    error_counts = df['consignor_website_error'].value_counts()
    if not error_counts.empty:
        print(f"\n" + "-"*60)
        print("ERROR TYPE BREAKDOWN")
        print("-"*60)
        for error, count in error_counts.head(5).items():
            if pd.notna(error):
                print(f"  {error}: {count:,} websites")
    
    # Contact info completion summary
    print(f"\n" + "="*60)
    print("CONTACT INFO COMPLETION SUMMARY")
    print("="*60)
    
    # Consignor stats
    consignor_stats = {
        'Total unique consignors': df['consignor_cleaned'].nunique(),
        'With websites': df['consignor_website'].notna().sum(),
        'With valid websites': df['consignor_website_valid'].sum(),
        'With emails': df['consignor_email'].notna().sum(),
        'With phones': df['consignor_phone'].notna().sum(),
        'With city info': df['consignor_city'].notna().sum(),
        'With state info': df['consignor_state'].notna().sum()
    }
    
    print(f"\nConsignor Contact Information:")
    for metric, count in consignor_stats.items():
        print(f"  {metric}: {count:,}")
    
    # Buyer stats
    buyer_stats = {
        'Total unique buyers': df['buyer_cleaned'].nunique(),
        'With websites': df['buyer_website'].notna().sum(),
        'With valid websites': df['buyer_website_valid'].sum(),
        'With emails': df['buyer_email'].notna().sum(),
        'With phones': df['buyer_phone'].notna().sum(),
        'With city info': df['buyer_city'].notna().sum(),
        'With state info': df['buyer_state'].notna().sum()
    }
    
    print(f"\nBuyer Contact Information:")
    for metric, count in buyer_stats.items():
        print(f"  {metric}: {count:,}")
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    analyze_validation_results('/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_validated.csv')