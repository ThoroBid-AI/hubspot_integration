import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_final_dataset():
    """Create the final cleaned and validated dataset"""
    
    # Read the validated data
    logger.info("Loading validated dataset...")
    df = pd.read_csv('/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_validated.csv')
    
    # Clean up column order for better readability
    final_columns = [
        # Core sales data
        'buyer', 'buyer_cleaned', 'consignor', 'consignor_cleaned',
        'sale_price', 'sale_year', 'source',
        
        # Consignor contact info
        'consignor_website', 'consignor_website_valid', 'consignor_website_status',
        'consignor_email', 'consignor_phone', 
        'consignor_city', 'consignor_state', 'consignor_country', 'consignor_is_company',
        
        # Buyer contact info  
        'buyer_website', 'buyer_website_valid', 'buyer_website_status',
        'buyer_email', 'buyer_phone',
        'buyer_city', 'buyer_state', 'buyer_country', 'buyer_is_company'
    ]
    
    # Select and reorder columns
    df_final = df[final_columns].copy()
    
    # Add some data quality indicators
    df_final['consignor_contact_score'] = (
        df_final['consignor_website_valid'].fillna(False).astype(int) +
        df_final['consignor_email'].notna().astype(int) +
        df_final['consignor_phone'].notna().astype(int) +
        df_final['consignor_city'].notna().astype(int)
    )
    
    df_final['buyer_contact_score'] = (
        df_final['buyer_website_valid'].fillna(False).astype(int) +
        df_final['buyer_email'].notna().astype(int) +
        df_final['buyer_phone'].notna().astype(int) +
        df_final['buyer_city'].notna().astype(int)
    )
    
    # Save the final dataset
    final_file = '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_enriched.csv'
    df_final.to_csv(final_file, index=False)
    logger.info(f"Final dataset saved to: {final_file}")
    
    # Print summary statistics
    logger.info(f"\nFinal Dataset Summary:")
    logger.info(f"Total records: {len(df_final):,}")
    logger.info(f"Date range: {df_final['sale_year'].min()} - {df_final['sale_year'].max()}")
    logger.info(f"Sources: {', '.join(df_final['source'].unique())}")
    logger.info(f"Unique consignors: {df_final['consignor_cleaned'].nunique():,}")
    logger.info(f"Unique buyers: {df_final['buyer_cleaned'].nunique():,}")
    
    # Contact info summary
    logger.info(f"\nConsignor Contact Info:")
    logger.info(f"  Valid websites: {df_final['consignor_website_valid'].sum():,}")
    logger.info(f"  Emails: {df_final['consignor_email'].notna().sum():,}")
    logger.info(f"  Phone numbers: {df_final['consignor_phone'].notna().sum():,}")
    
    logger.info(f"\nBuyer Contact Info:")
    logger.info(f"  Valid websites: {df_final['buyer_website_valid'].sum():,}")
    logger.info(f"  Emails: {df_final['buyer_email'].notna().sum():,}")
    logger.info(f"  Phone numbers: {df_final['buyer_phone'].notna().sum():,}")
    
    return final_file

def cleanup_old_files():
    """Remove intermediate files"""
    files_to_remove = [
        '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/combined_sales_data.csv',
        '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_v2.csv',
        '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_v3.csv',
        '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_final_v4.csv',
        '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_combined_clean.csv',
        '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_cleaned.csv',
        '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_enriched.csv',
        '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_enriched_v2.csv',
        '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/horse_sales_validated.csv',
        '/Users/maxwittenberg/Desktop/Sales_Buyer_Consignor/test_export.csv'
    ]
    
    logger.info("\nCleaning up intermediate files...")
    removed_count = 0
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"  Removed: {os.path.basename(file_path)}")
                removed_count += 1
            except Exception as e:
                logger.error(f"  Error removing {os.path.basename(file_path)}: {e}")
        else:
            logger.info(f"  Not found: {os.path.basename(file_path)}")
    
    logger.info(f"\nRemoved {removed_count} intermediate files")

if __name__ == "__main__":
    # Create final dataset
    final_file = create_final_dataset()
    
    # Clean up old files
    cleanup_old_files()
    
    logger.info(f"\n{'='*60}")
    logger.info("PROCESS COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Final dataset: {os.path.basename(final_file)}")
    logger.info(f"Location: {os.path.dirname(final_file)}")
    logger.info(f"{'='*60}")