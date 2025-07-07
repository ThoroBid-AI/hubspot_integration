import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_recent_years():
    """Analyze unique buyers and consignors from the past 3 years"""
    
    # Load the final dataset
    logger.info("Loading final dataset...")
    df = pd.read_csv('/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_enriched.csv')
    
    # Get current year and calculate 3-year window
    current_year = datetime.now().year
    past_3_years = [current_year - 2, current_year - 1, current_year]  # 2023, 2024, 2025
    
    logger.info(f"Current year: {current_year}")
    logger.info(f"Analyzing past 3 years: {past_3_years}")
    
    # Filter data for past 3 years
    recent_data = df[df['sale_year'].isin(past_3_years)]
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS: PAST 3 YEARS ({min(past_3_years)}-{max(past_3_years)})")
    print(f"{'='*60}")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total records (past 3 years): {len(recent_data):,}")
    print(f"  Total records (all years): {len(df):,}")
    print(f"  Percentage of recent data: {len(recent_data)/len(df)*100:.1f}%")
    
    # Year breakdown
    print(f"\nRecords by Year:")
    year_counts = recent_data['sale_year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {int(year)}: {count:,} records")
    
    # Source breakdown for recent years
    print(f"\nRecords by Source (past 3 years):")
    source_counts = recent_data['source'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count:,} records")
    
    # Unique entities analysis
    print(f"\n{'='*60}")
    print(f"UNIQUE ENTITIES ANALYSIS")
    print(f"{'='*60}")
    
    # Consignors
    unique_consignors_recent = recent_data['consignor_cleaned'].nunique()
    unique_consignors_all = df['consignor_cleaned'].nunique()
    
    print(f"\nConsignors:")
    print(f"  Unique consignors (past 3 years): {unique_consignors_recent:,}")
    print(f"  Unique consignors (all time): {unique_consignors_all:,}")
    print(f"  Recent as % of all-time: {unique_consignors_recent/unique_consignors_all*100:.1f}%")
    
    # Buyers
    unique_buyers_recent = recent_data['buyer_cleaned'].dropna().nunique()
    unique_buyers_all = df['buyer_cleaned'].dropna().nunique()
    
    print(f"\nBuyers:")
    print(f"  Unique buyers (past 3 years): {unique_buyers_recent:,}")
    print(f"  Unique buyers (all time): {unique_buyers_all:,}")
    print(f"  Recent as % of all-time: {unique_buyers_recent/unique_buyers_all*100:.1f}%")
    
    # Top consignors in recent years
    print(f"\n{'='*60}")
    print(f"TOP 10 CONSIGNORS (PAST 3 YEARS)")
    print(f"{'='*60}")
    
    top_consignors = recent_data['consignor_cleaned'].value_counts().head(10)
    for i, (consignor, count) in enumerate(top_consignors.items(), 1):
        print(f"  {i:2d}. {consignor}: {count:,} sales")
    
    # Top buyers in recent years (excluding NOT SOLD, OUT, etc.)
    print(f"\n{'='*60}")
    print(f"TOP 10 BUYERS (PAST 3 YEARS)")
    print(f"{'='*60}")
    
    # Filter out non-buyers
    actual_buyers = recent_data[
        recent_data['buyer_cleaned'].notna() & 
        ~recent_data['buyer_cleaned'].isin(['NOT SOLD', 'OUT', 'WITHDRAWN', ''])
    ]
    
    top_buyers = actual_buyers['buyer_cleaned'].value_counts().head(10)
    for i, (buyer, count) in enumerate(top_buyers.items(), 1):
        print(f"  {i:2d}. {buyer}: {count:,} purchases")
    
    # Contact info completeness for recent entities
    print(f"\n{'='*60}")
    print(f"CONTACT INFO COMPLETENESS (PAST 3 YEARS)")
    print(f"{'='*60}")
    
    # Recent consignors with contact info
    recent_consignors_with_websites = recent_data['consignor_website_valid'].sum()
    recent_consignors_with_emails = recent_data['consignor_email'].notna().sum()
    recent_consignors_with_phones = recent_data['consignor_phone'].notna().sum()
    
    print(f"\nConsignor Contact Info (records with data):")
    print(f"  Valid websites: {recent_consignors_with_websites:,}")
    print(f"  Email addresses: {recent_consignors_with_emails:,}")
    print(f"  Phone numbers: {recent_consignors_with_phones:,}")
    
    # Sales volume analysis
    print(f"\n{'='*60}")
    print(f"SALES VOLUME ANALYSIS (PAST 3 YEARS)")
    print(f"{'='*60}")
    
    # Remove records without prices
    recent_with_prices = recent_data[recent_data['sale_price'].notna() & (recent_data['sale_price'] > 0)]
    
    if len(recent_with_prices) > 0:
        total_sales_value = recent_with_prices['sale_price'].sum()
        avg_sale_price = recent_with_prices['sale_price'].mean()
        median_sale_price = recent_with_prices['sale_price'].median()
        
        print(f"\nSales with Prices:")
        print(f"  Total records with prices: {len(recent_with_prices):,}")
        print(f"  Total sales value: ${total_sales_value:,.0f}")
        print(f"  Average sale price: ${avg_sale_price:,.0f}")
        print(f"  Median sale price: ${median_sale_price:,.0f}")
        
        # Price ranges
        price_ranges = [
            ("Under $10K", recent_with_prices[recent_with_prices['sale_price'] < 10000]),
            ("$10K - $50K", recent_with_prices[(recent_with_prices['sale_price'] >= 10000) & (recent_with_prices['sale_price'] < 50000)]),
            ("$50K - $100K", recent_with_prices[(recent_with_prices['sale_price'] >= 50000) & (recent_with_prices['sale_price'] < 100000)]),
            ("$100K+", recent_with_prices[recent_with_prices['sale_price'] >= 100000])
        ]
        
        print(f"\nPrice Distribution:")
        for range_name, range_data in price_ranges:
            count = len(range_data)
            pct = count / len(recent_with_prices) * 100
            print(f"  {range_name}: {count:,} sales ({pct:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    analyze_recent_years()