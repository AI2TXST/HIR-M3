import os
import re
import time
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, ByteType, DoubleType, StringType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_spark_session(app_name="OASIS_Preprocessing_NB_Condensed"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "16g") \
        .config("spark.driver.maxResultSize", "16g") \
        .config("spark.sql.codegen.wholeStage", "false") \
        .config("spark.sql.codegen.maxFields", "100") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def preprocess_notebook_pipeline_condensed(
    input_csv, 
    output_csv, 
    output_parquet=None,
    min_freq_count=200, 
    use_frequency_encoding_for_ids=True
):
    """
    Optimized and Condensed PySpark Preprocessing Pipeline for OASIS Beneficiary Data.
    
    Key Memory & Size Optimizations:
    1. min_freq_count=200: Collapses rare categories into _Other, reducing total columns from 6,600+ down to ~600.
    2. Frequency Encoding for IDs: Replaces thousands of high-cardinality agency/facility dummy columns 
       with 2 dense frequency feature columns.
    3. ByteType Casts: Stores one-hot dummy flags as 1-byte integers instead of 8-byte doubles.
    4. Dual Export: Saves snappy-compressed Parquet (.parquet) alongside condensed CSV.
    """
    spark = get_spark_session()
    
    logging.info(f"Loading data from {input_csv} via PySpark...")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_csv)
    
    total_rows = df.count()
    total_cols = len(df.columns)
    logging.info(f"Initial PySpark shape: ({total_rows}, {total_cols})")

    # 1. Clean facility and agency identifiers
    id_cols = ["Facility_Internal_ID", "Agency_Medicare_Number"]
    for col_name in id_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
            df = df.filter(F.col(col_name).isNotNull() & ~F.isnan(F.col(col_name)))
            df = df.withColumn(col_name, F.col(col_name).cast("long").cast("string"))

            # Frequency Encoding: Replace high-cardinality IDs with admission count frequency
            if use_frequency_encoding_for_ids:
                freq_table = df.groupBy(col_name).agg(F.count("*").alias(f"{col_name}_freq"))
                df = df.join(freq_table, on=col_name, how="left")

    # 2. Define numeric and binary flag columns
    numeric_cols = [
        'Age', 'Days_Cared_For', 'charlson_score', 'charlson_ageadj',
        'charlson_survival_10yr', 'elix_quan_score', 'elix_quan_ageadj',
        'elix_swiss_score', 'elix_swiss_ageadj', 'POP_URB', 'POPPCT_URB',
        'POP_RUR', 'POPPCT_RUR', 'ACS_PCT_BACHELOR_DGR', 'ACS_PCT_COLLEGE_ASSOCIATE_DGR',
        'ACS_PCT_GRADUATE_DGR', 'ACS_PCT_HS_GRADUATE', 'ACS_PCT_LT_HS',
        'ACS_PCT_NO_WORK_NO_SCHL_16_19', 'ACS_PCT_POSTHS_ED', 'ACS_PCT_VET_BACHELOR',
        'ACS_PCT_VET_COLLEGE', 'ACS_PCT_VET_HS', 'ACS_PCT_HH_LIMIT_ENGLISH',
        'ACS_PCT_HH_BROADBAND', 'ACS_PCT_HH_BROADBAND_ONLY', 'ACS_PCT_HH_CELLULAR',
        'ACS_PCT_HH_CELLULAR_ONLY', 'ACS_PCT_HH_DIAL_INTERNET_ONLY',
        'ACS_PCT_HH_INTERNET', 'ACS_PCT_HH_INTERNET_NO_SUBS',
        'ACS_PCT_HH_NO_COMP_DEV', 'ACS_PCT_HH_NO_INTERNET', 'ACS_PCT_HH_OTHER_COMP',
        'ACS_PCT_HH_OTHER_COMP_ONLY', 'ACS_PCT_HH_PC', 'ACS_PCT_HH_PC_ONLY',
        'ACS_PCT_HH_SAT_INTERNET', 'ACS_PCT_HH_SMARTPHONE', 'ACS_PCT_HH_SMARTPHONE_ONLY',
        'ACS_PCT_HH_TABLET', 'ACS_PCT_HH_TABLET_ONLY', 'ACS_PCT_CHILDREN_GRANDPARENT',
        'ACS_PCT_CHILD_1FAM', 'ACS_PCT_GRANDP_NO_RESPS', 'ACS_PCT_GRANDP_RESPS_NO_P',
        'ACS_PCT_GRANDP_RESPS_P', 'ACS_PCT_HH_1PERS', 'ACS_PCT_HH_ABOVE65',
        'ACS_PCT_HH_ALONE_ABOVE65', 'ACS_PCT_HH_KID_1PRNT', 'ACS_TOT_GRANDCHILDREN_GP',
        'ACS_PCT_HEALTH_INC_138_199', 'ACS_PCT_HEALTH_INC_200_399',
        'ACS_PCT_HEALTH_INC_ABOVE400', 'ACS_PCT_HEALTH_INC_BELOW137',
        'ACS_PCT_HH_1FAM_FOOD_STMP', 'ACS_PCT_HH_FOOD_STMP_BLW_POV',
        'ACS_PCT_HH_NO_FD_STMP_BLW_POV', 'ACS_PCT_HH_PUB_ASSIST', 'ACS_PCT_INC50',
        'ACS_PCT_INC50_ABOVE65', 'ACS_PCT_NONVET_POV_18_64', 'ACS_PCT_PERSON_INC_100_124',
        'ACS_PCT_PERSON_INC_125_199', 'ACS_PCT_PERSON_INC_ABOVE200',
        'ACS_PCT_PERSON_INC_BELOW99', 'ACS_PCT_POV_AIAN', 'ACS_PCT_POV_ASIAN',
        'ACS_PCT_POV_BLACK', 'ACS_PCT_POV_HISPANIC', 'ACS_PCT_POV_MULTI',
        'ACS_PCT_POV_NHPI', 'ACS_PCT_POV_OTHER', 'ACS_PCT_POV_WHITE',
        'ACS_PCT_VET_POV_18_64', 'ACS_TOT_POP_POV'
    ]

    # Coerce numeric columns to FloatType (4 bytes) & round
    num_cols_present = [c for c in numeric_cols if c in df.columns]
    for c in num_cols_present:
        df = df.withColumn(c, F.col(c).cast(FloatType()))
        df = df.filter(F.col(c).isNotNull() & ~F.isnan(F.col(c)))
        df = df.withColumn(c, F.round(F.col(c), 2))

    # 3. Drop specified metadata/text and unwanted comorbidity columns
    drop_cols = [
        'zip', 'city', 'state_name', 'COUNTYFIPS', 'BMI',
        'Primary_Diagnosis_ICD_10_C_M_Code_Cluster_Name',
        'Other_Diagnosis_Code_1_ICD_10_C_M_Cluster_Name',
        'Other_Diagnosis_Code_2_ICD_10_C_M_Cluster_Name',
        'Other_Diagnosis_Code_3_ICD_10_C_M_Cluster_Name',
        'Other_Diagnosis_Code_4_ICD_10_C_M_Cluster_Name',
        'Other_Diagnosis_Code_5_ICD_10_C_M_Cluster_Name',
        'ICD_Clusters_In_Row', 'Assessment_Effective_Date',
        'ICD_Cluster_Names_In_Row',
        'blane', 'diab', 'diabc', 'diabunc', 'diabwc', 'msld', 'pud'
    ]
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(*cols_to_drop)

    # 4. One-hot encode categorical columns with min_freq_count threshold
    cat_cols = ['Submitted_HIPPS_Code', 'Gender', 'ByDiscipline', 'BMI_Category', 'COUNTY_NAME']
    if not use_frequency_encoding_for_ids:
        cat_cols.extend(id_cols)

    icd_cols = [c for c in df.columns if ('ICD' in c or 'Diagnosis' in c) and c not in drop_cols]
    all_cat_cols = list(dict.fromkeys(cat_cols + icd_cols))
    present_cat_cols = [c for c in all_cat_cols if c in df.columns]

    for col_name in present_cat_cols:
        df = df.withColumn(col_name, F.regexp_replace(F.trim(F.col(col_name).cast("string")), r"\.0$", ""))
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name).isin("nan", "None", "<NA>", ""), None).otherwise(F.col(col_name))
        )

    if present_cat_cols:
        logging.info(f"One-hot encoding {len(present_cat_cols)} categorical columns (min_freq_count={min_freq_count})...")
        dummy_exprs = []
        cols_to_drop_cat = []
        seen_dummy_names = set(df.columns)

        for c in present_cat_cols:
            freq_df = df.groupBy(f"`{c}`").count().collect()
            
            frequent_vals = []
            has_rare = False
            for row in freq_df:
                val = row[c]
                cnt = row['count']
                if val is not None:
                    if cnt >= min_freq_count:
                        frequent_vals.append(str(val))
                    else:
                        has_rare = True

            all_vals = list(set([str(r[c]) for r in freq_df if r[c] is not None]))
            all_vals.sort()
            first_val = all_vals[0] if len(all_vals) > 0 else None

            frequent_vals.sort()
            for val in frequent_vals:
                if val == first_val:
                    continue  # drop_first
                clean_v = re.sub(r'[^a-zA-Z0-9]+', '_', str(val)).strip('_')
                col_dummy_name = f"{c}_{clean_v}"
                
                if col_dummy_name in seen_dummy_names:
                    col_dummy_name = f"{c}_val_{clean_v}"
                
                if col_dummy_name not in seen_dummy_names:
                    seen_dummy_names.add(col_dummy_name)
                    dummy_exprs.append(
                        F.when(F.col(f"`{c}`") == val, F.lit(1).cast(ByteType())).otherwise(F.lit(0).cast(ByteType())).alias(col_dummy_name)
                    )

            if has_rare and 'Other' != first_val:
                col_dummy_name = f"{c}_Other"
                if col_dummy_name not in seen_dummy_names:
                    seen_dummy_names.add(col_dummy_name)
                    dummy_exprs.append(
                        F.when(~F.col(f"`{c}`").isin(frequent_vals) & F.col(f"`{c}`").isNotNull(), F.lit(1).cast(ByteType())).otherwise(F.lit(0).cast(ByteType())).alias(col_dummy_name)
                    )

            cols_to_drop_cat.append(c)

        if dummy_exprs:
            df = df.select("*", *dummy_exprs)
        if cols_to_drop_cat:
            df = df.drop(*cols_to_drop_cat)

    # Clean column names
    for old_col in df.columns:
        new_col = re.sub(r'[^a-zA-Z0-9]+', '_', old_col).strip('_')
        if old_col != new_col:
            df = df.withColumnRenamed(old_col, new_col)

    # Deduplicate column names
    unique_cols = []
    seen = set()
    for col_name in df.columns:
        if col_name not in seen:
            seen.add(col_name)
            unique_cols.append(col_name)

    df = df.select(*[F.col(f"`{c}`") for c in unique_cols])

    # Filter null/NaN rows for Beneficiary_ID
    for bene_col in ["Beneficiary_ID", "BENE_ID", "beneficiary_id"]:
        if bene_col in df.columns:
            logging.info(f"Filtering out null/NaN rows for {bene_col}...")
            df = df.filter(
                F.col(f"`{bene_col}`").isNotNull() & 
                ~F.isnan(F.col(f"`{bene_col}`")) & 
                (F.col(f"`{bene_col}`") != "") & 
                (F.col(f"`{bene_col}`") != "nan")
            )

    # 5. Batched PySpark constant column check
    logging.info("Checking for constant columns in PySpark...")
    cols = df.columns
    batch_size = 100
    constant_cols = []
    for i in range(0, len(cols), batch_size):
        batch = cols[i:i+batch_size]
        exprs = [F.countDistinct(F.col(f"`{c}`")).alias(c) for c in batch]
        summary_batch = df.agg(*exprs).collect()[0]
        for c in batch:
            if summary_batch[c] <= 1:
                constant_cols.append(c)

    if constant_cols:
        logging.info(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
        df = df.drop(*constant_cols)

    final_shape = (df.count(), len(df.columns))
    logging.info(f"Condensed PySpark final shape: {final_shape}")

    # 6a. Save snappy-compressed Parquet output if requested
    if not output_parquet:
        output_parquet = output_csv.replace(".csv", ".parquet")

    logging.info(f"Saving snappy-compressed Parquet dataset to {output_parquet}...")
    df.write.mode("overwrite").parquet(output_parquet)
    logging.info(f"Successfully saved Parquet dataset to {output_parquet}!")

    # 6b. Save condensed CSV dataset directly from PySpark
    logging.info(f"Saving condensed CSV dataset to {output_csv}...")
    temp_dir = output_csv + "_temp_spark"
    df.coalesce(1).write.option("header", "true").mode("overwrite").csv(temp_dir)

    import glob
    import shutil
    part_files = glob.glob(os.path.join(temp_dir, "part-*.csv"))
    if part_files:
        if os.path.exists(output_csv):
            os.remove(output_csv)
        shutil.move(part_files[0], output_csv)
        shutil.rmtree(temp_dir, ignore_errors=True)
        logging.info(f"Successfully saved condensed CSV to {output_csv}!")
    else:
        logging.error("Failed to locate PySpark part-*.csv output file!")

    spark.stop()

def main():
    start_time = time.time()
    input_file = "data/final_mergedDF.csv"
    output_file = "data/processed_final_mergedDF_condensed.csv"
    
    preprocess_notebook_pipeline_condensed(input_file, output_file, min_freq_count=200)
    logging.info(f"Total time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
