#!/usr/bin/env python3
"""
Analyze and map columns across all three BigQuery tables for union
"""

# Define all columns from each table
ocala_columns = [
    'hip_number', 'horse_name', 'color', 'sex', 'foaling_date', 'foaling_year', 
    'foaling_month', 'foaling_day', 'sire_name', 'dam_name', 'dam_sire', 
    'consignor', 'foaling_area', 'barn_number', 'buyer_name', 'sale_price', 
    'reserve_price', 'private_sale_indicator', 'under_tack_time', 'work_set', 
    'work_day', 'in_foal_sire', 'last_bred', 'in_out_status', 'ut_distance', 
    'ut_actual_date', 'ut_group', 'ut_set', 'pp_pdf_link', 'source_file', 
    'source_year', 'sale_type'
]

keeneland_columns = [
    'dd', 'color', 'consignor_central', 'grand_sire', 'session', 'book', 
    'dob_formatted', 'sold_as', 'last_service_date', 'yob', 'bceligible', 
    'hip', 'dam', 'sale_price_formatted', 'rna_indicator', 'barn', 'consignor', 
    'stakes_producing_dam', 'pp_file_name', 'state_foaled', 'currentsale', 
    'sex', 'stakes_producer', 'salesupdate', 'pregnancy_status', 'updates_link', 
    'broodmare_sire', 'out_indicator', 'sale_price', 'catalog_page_link', 
    'covering_sire_sire', 'buyer', 'lastshowdate', 'search_id', 'covering_sire', 
    'sale', 'ok2show', 'keecatalog_saleid', 'dob', 'grade', 'name', 'comment', 
    'sire', 'kee_bid_link', 'show_pedigree_indicator', 'dam_yob'
]

fasigtipton_columns = [
    'SALE_TITLE', 'SESSION', 'HIP', 'COLOR', 'NAME', 'SEX', 'SIRE', 'DAM', 
    'SIRE_OF_DAM', 'PROPERTY_LINE', 'PURCHASER', 'PRICE', 'PRIVATE_SALE', 
    'YEAR_OF_BIRTH', 'FOALED', 'BARN', 'COVERING_SIRE', 'COVER_DATE', 
    'CONSIGNOR_NAME', 'SOLD_AS_CODE', 'SOLD_AS_DESCRIPTION', 'VIRTUAL_INSPECTION'
]

# Create unified column mapping
column_mapping = {
    # Core horse information
    'horse_name': {
        'ocala': 'horse_name',
        'keeneland': 'name', 
        'fasigtipton': 'NAME'
    },
    'hip_number': {
        'ocala': 'hip_number',
        'keeneland': 'hip',
        'fasigtipton': 'HIP'
    },
    'color': {
        'ocala': 'color',
        'keeneland': 'color',
        'fasigtipton': 'COLOR'
    },
    'sex': {
        'ocala': 'sex',
        'keeneland': 'sex',
        'fasigtipton': 'SEX'
    },
    'sire_name': {
        'ocala': 'sire_name',
        'keeneland': 'sire',
        'fasigtipton': 'SIRE'
    },
    'dam_name': {
        'ocala': 'dam_name',
        'keeneland': 'dam',
        'fasigtipton': 'DAM'
    },
    'dam_sire': {
        'ocala': 'dam_sire',
        'keeneland': 'broodmare_sire',
        'fasigtipton': 'SIRE_OF_DAM'
    },
    
    # Sale information
    'consignor': {
        'ocala': 'consignor',
        'keeneland': 'consignor',
        'fasigtipton': 'CONSIGNOR_NAME'
    },
    'buyer_name': {
        'ocala': 'buyer_name',
        'keeneland': 'buyer',
        'fasigtipton': 'PURCHASER'
    },
    'sale_price': {
        'ocala': 'sale_price',
        'keeneland': 'sale_price',
        'fasigtipton': 'PRICE'
    },
    'barn_number': {
        'ocala': 'barn_number',
        'keeneland': 'barn',
        'fasigtipton': 'BARN'
    },
    
    # Date information
    'foaling_date': {
        'ocala': 'foaling_date',
        'keeneland': 'dob_formatted',
        'fasigtipton': 'NULL'
    },
    'foaling_year': {
        'ocala': 'foaling_year',
        'keeneland': 'yob',
        'fasigtipton': 'YEAR_OF_BIRTH'
    },
    'birth_date': {
        'ocala': 'foaling_date',
        'keeneland': 'dob',
        'fasigtipton': 'NULL'
    },
    
    # Sale-specific fields
    'sale_title': {
        'ocala': 'sale_type',
        'keeneland': 'sale',
        'fasigtipton': 'SALE_TITLE'
    },
    'session': {
        'ocala': 'NULL',
        'keeneland': 'session',
        'fasigtipton': 'SESSION'
    },
    'reserve_price': {
        'ocala': 'reserve_price',
        'keeneland': 'NULL',
        'fasigtipton': 'NULL'
    },
    'private_sale_indicator': {
        'ocala': 'private_sale_indicator',
        'keeneland': 'NULL',
        'fasigtipton': 'PRIVATE_SALE'
    },
    
    # Additional breeding info
    'covering_sire': {
        'ocala': 'in_foal_sire',
        'keeneland': 'covering_sire',
        'fasigtipton': 'COVERING_SIRE'
    },
    'last_bred_date': {
        'ocala': 'last_bred',
        'keeneland': 'last_service_date',
        'fasigtipton': 'COVER_DATE'
    },
    
    # Performance data (Ocala-specific)
    'under_tack_time': {
        'ocala': 'under_tack_time',
        'keeneland': 'NULL',
        'fasigtipton': 'NULL'
    },
    'ut_distance': {
        'ocala': 'ut_distance',
        'keeneland': 'NULL',
        'fasigtipton': 'NULL'
    },
    
    # Status fields
    'rna_indicator': {
        'ocala': 'NULL',
        'keeneland': 'rna_indicator',
        'fasigtipton': 'NULL'
    },
    'out_indicator': {
        'ocala': 'in_out_status',
        'keeneland': 'out_indicator',
        'fasigtipton': 'NULL'
    },
    
    # Geographic info
    'state_foaled': {
        'ocala': 'foaling_area',
        'keeneland': 'state_foaled',
        'fasigtipton': 'FOALED'
    },
    
    # Source tracking
    'source_file': {
        'ocala': 'source_file',
        'keeneland': 'pp_file_name',
        'fasigtipton': 'NULL'
    },
    'source_year': {
        'ocala': 'source_year',
        'keeneland': 'NULL',
        'fasigtipton': 'NULL'
    }
}

def generate_union_query():
    """Generate the complete UNION query with all columns mapped"""
    
    # Get all unique column names
    all_columns = list(column_mapping.keys())
    
    print("=== COMPLETE UNION QUERY FOR ALL THREE TABLES ===\n")
    
    query = "SELECT\n"
    
    # Add each column with proper casting and aliasing
    for i, col in enumerate(all_columns):
        comma = "," if i < len(all_columns) - 1 else ""
        
        # Ocala mapping
        ocala_col = column_mapping[col]['ocala']
        if ocala_col == 'NULL':
            ocala_val = "NULL"
        else:
            ocala_val = f"CAST({ocala_col} AS STRING)"
        
        # Keeneland mapping  
        keeneland_col = column_mapping[col]['keeneland']
        if keeneland_col == 'NULL':
            keeneland_val = "NULL"
        else:
            keeneland_val = f"CAST({keeneland_col} AS STRING)"
        
        # Fasigtipton mapping
        fasigtipton_col = column_mapping[col]['fasigtipton'] 
        if fasigtipton_col == 'NULL':
            fasigtipton_val = "NULL"
        else:
            fasigtipton_val = f"CAST({fasigtipton_col} AS STRING)"
            
        query += f"  {col}{comma}\n"
    
    query += "FROM (\n"
    
    # Ocala subquery
    query += "  -- Ocala Sales\n  SELECT\n"
    for i, col in enumerate(all_columns):
        comma = "," if i < len(all_columns) - 1 else ""
        ocala_col = column_mapping[col]['ocala']
        if ocala_col == 'NULL':
            ocala_val = "NULL"
        else:
            ocala_val = f"CAST({ocala_col} AS STRING)"
        query += f"    {ocala_val} as {col}{comma}\n"
    query += "    'ocala' as source\n"
    query += "  FROM `thorobid-dev.ingest.ocala_sales`\n"
    query += "  WHERE foaling_year >= 1990.0\n"
    query += "  AND consignor IS NOT NULL\n"
    query += "  AND buyer_name IS NOT NULL\n\n"
    
    query += "  UNION ALL\n\n"
    
    # Keeneland subquery
    query += "  -- Keeneland Sales\n  SELECT\n"
    for i, col in enumerate(all_columns):
        comma = "," if i < len(all_columns) - 1 else ""
        keeneland_col = column_mapping[col]['keeneland']
        if keeneland_col == 'NULL':
            keeneland_val = "NULL"
        else:
            keeneland_val = f"CAST({keeneland_col} AS STRING)"
        query += f"    {keeneland_val} as {col}{comma}\n"
    query += "    'keeneland' as source\n"
    query += "  FROM `thorobid-dev.ingest.keeneland_sales`\n"
    query += "  WHERE COALESCE(EXTRACT(YEAR FROM dob_formatted), EXTRACT(YEAR FROM dob)) >= 1990\n"
    query += "  AND consignor IS NOT NULL\n"
    query += "  AND buyer IS NOT NULL\n\n"
    
    query += "  UNION ALL\n\n"
    
    # Fasigtipton subquery
    query += "  -- Fasig-Tipton Sales\n  SELECT\n"
    for i, col in enumerate(all_columns):
        comma = "," if i < len(all_columns) - 1 else ""
        fasigtipton_col = column_mapping[col]['fasigtipton']
        if fasigtipton_col == 'NULL':
            fasigtipton_val = "NULL"
        else:
            fasigtipton_val = f"CAST({fasigtipton_col} AS STRING)"
        query += f"    {fasigtipton_val} as {col}{comma}\n"
    query += "    'fasigtipton' as source\n"
    query += "  FROM `thorobid-dev.ingest.fasigtipton_sales`\n"
    query += "  WHERE YEAR_OF_BIRTH >= 1990\n"
    query += "  AND CONSIGNOR_NAME IS NOT NULL\n"
    query += "  AND PURCHASER IS NOT NULL\n\n"
    
    query += ")\nORDER BY foaling_year DESC"
    
    return query

if __name__ == "__main__":
    union_query = generate_union_query()
    print(union_query)
    
    # Save to file
    with open('complete_union_query.sql', 'w') as f:
        f.write(union_query)
    
    print(f"\n\nQuery saved to: complete_union_query.sql")
    print(f"Total unified columns: {len(column_mapping)}")