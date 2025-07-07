SELECT
  horse_name,
  hip_number,
  color,
  sex,
  sire_name,
  dam_name,
  dam_sire,
  consignor,
  buyer_name,
  sale_price,
  barn_number,
  foaling_date,
  foaling_year,
  birth_date,
  sale_title,
  session,
  reserve_price,
  private_sale_indicator,
  covering_sire,
  last_bred_date,
  under_tack_time,
  ut_distance,
  rna_indicator,
  out_indicator,
  state_foaled,
  source_file,
  source_year
FROM (
  -- Ocala Sales
  SELECT
    CAST(horse_name AS STRING) as horse_name,
    CAST(hip_number AS STRING) as hip_number,
    CAST(color AS STRING) as color,
    CAST(sex AS STRING) as sex,
    CAST(sire_name AS STRING) as sire_name,
    CAST(dam_name AS STRING) as dam_name,
    CAST(dam_sire AS STRING) as dam_sire,
    CAST(consignor AS STRING) as consignor,
    CAST(buyer_name AS STRING) as buyer_name,
    CAST(sale_price AS STRING) as sale_price,
    CAST(barn_number AS STRING) as barn_number,
    CAST(foaling_date AS STRING) as foaling_date,
    CAST(foaling_year AS STRING) as foaling_year,
    CAST(foaling_date AS STRING) as birth_date,
    CAST(sale_type AS STRING) as sale_title,
    NULL as session,
    CAST(reserve_price AS STRING) as reserve_price,
    CAST(private_sale_indicator AS STRING) as private_sale_indicator,
    CAST(in_foal_sire AS STRING) as covering_sire,
    CAST(last_bred AS STRING) as last_bred_date,
    CAST(under_tack_time AS STRING) as under_tack_time,
    CAST(ut_distance AS STRING) as ut_distance,
    NULL as rna_indicator,
    CAST(in_out_status AS STRING) as out_indicator,
    CAST(foaling_area AS STRING) as state_foaled,
    CAST(source_file AS STRING) as source_file,
    CAST(source_year AS STRING) as source_year
    'ocala' as source
  FROM `thorobid-dev.ingest.ocala_sales`
  WHERE foaling_year >= 1990.0
  AND consignor IS NOT NULL
  AND buyer_name IS NOT NULL

  UNION ALL

  -- Keeneland Sales
  SELECT
    CAST(name AS STRING) as horse_name,
    CAST(hip AS STRING) as hip_number,
    CAST(color AS STRING) as color,
    CAST(sex AS STRING) as sex,
    CAST(sire AS STRING) as sire_name,
    CAST(dam AS STRING) as dam_name,
    CAST(broodmare_sire AS STRING) as dam_sire,
    CAST(consignor AS STRING) as consignor,
    CAST(buyer AS STRING) as buyer_name,
    CAST(sale_price AS STRING) as sale_price,
    CAST(barn AS STRING) as barn_number,
    CAST(dob_formatted AS STRING) as foaling_date,
    CAST(yob AS STRING) as foaling_year,
    CAST(dob AS STRING) as birth_date,
    CAST(sale AS STRING) as sale_title,
    CAST(session AS STRING) as session,
    NULL as reserve_price,
    NULL as private_sale_indicator,
    CAST(covering_sire AS STRING) as covering_sire,
    CAST(last_service_date AS STRING) as last_bred_date,
    NULL as under_tack_time,
    NULL as ut_distance,
    CAST(rna_indicator AS STRING) as rna_indicator,
    CAST(out_indicator AS STRING) as out_indicator,
    CAST(state_foaled AS STRING) as state_foaled,
    CAST(pp_file_name AS STRING) as source_file,
    NULL as source_year
    'keeneland' as source
  FROM `thorobid-dev.ingest.keeneland_sales`
  WHERE COALESCE(EXTRACT(YEAR FROM dob_formatted), EXTRACT(YEAR FROM dob)) >= 1990
  AND consignor IS NOT NULL
  AND buyer IS NOT NULL

  UNION ALL

  -- Fasig-Tipton Sales
  SELECT
    CAST(NAME AS STRING) as horse_name,
    CAST(HIP AS STRING) as hip_number,
    CAST(COLOR AS STRING) as color,
    CAST(SEX AS STRING) as sex,
    CAST(SIRE AS STRING) as sire_name,
    CAST(DAM AS STRING) as dam_name,
    CAST(SIRE_OF_DAM AS STRING) as dam_sire,
    CAST(CONSIGNOR_NAME AS STRING) as consignor,
    CAST(PURCHASER AS STRING) as buyer_name,
    CAST(PRICE AS STRING) as sale_price,
    CAST(BARN AS STRING) as barn_number,
    NULL as foaling_date,
    CAST(YEAR_OF_BIRTH AS STRING) as foaling_year,
    NULL as birth_date,
    CAST(SALE_TITLE AS STRING) as sale_title,
    CAST(SESSION AS STRING) as session,
    NULL as reserve_price,
    CAST(PRIVATE_SALE AS STRING) as private_sale_indicator,
    CAST(COVERING_SIRE AS STRING) as covering_sire,
    CAST(COVER_DATE AS STRING) as last_bred_date,
    NULL as under_tack_time,
    NULL as ut_distance,
    NULL as rna_indicator,
    NULL as out_indicator,
    CAST(FOALED AS STRING) as state_foaled,
    NULL as source_file,
    NULL as source_year
    'fasigtipton' as source
  FROM `thorobid-dev.ingest.fasigtipton_sales`
  WHERE YEAR_OF_BIRTH >= 1990
  AND CONSIGNOR_NAME IS NOT NULL
  AND PURCHASER IS NOT NULL

)
ORDER BY foaling_year DESC