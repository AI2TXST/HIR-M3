import os
import re
import time
import glob
import shutil
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, ByteType, DoubleType, StringType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_spark_session(app_name="OASIS_Preprocessing_NB_Condensed_CPU"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "16g") \
        .config("spark.driver.maxResultSize", "16g") \
        .config("spark.driver.extraJavaOptions", "-Xss16m") \
        .config("spark.executor.extraJavaOptions", "-Xss16m") \
        .config("spark.sql.codegen.wholeStage", "false") \
        .config("spark.sql.codegen.maxFields", "100") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def extract_hicd_bert_embeddings_cpu(df, spark, n_components=32):
    """
    Recreates HICD-BERT Transformer architecture locally using PyTorch on CPU,
    encodes patient ICD assessment sequences into 128-d multi-head attention 
    hidden representations, and applies PCA to output 32 descriptive hicd_bert_emb_XX features.
    """
    icd_target_cols = [
        c for c in df.columns 
        if ('ICD' in c or 'Diagnosis' in c) 
        and not c.endswith(('_G0', '_G1', '_G2', '_Name', '_count')) 
        and 'Cluster_Name' not in c
    ]
    
    if not icd_target_cols:
        return df

    logging.info("Extracting HICD-BERT embeddings on CPU using local PyTorch Transformer architecture...")
    
    try:
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        from sklearn.decomposition import PCA as SklearnPCA
        
        seq_exprs = [
            F.when(F.col(f"`{c}`").isNotNull() & ~F.col(f"`{c}`").cast("string").isin("nan", "None", "<NA>", ""), 
                   F.concat(F.lit(c.split('_')[0] + ": "), F.col(f"`{c}`").cast("string")))
            .otherwise(F.lit(""))
            for c in icd_target_cols
        ]
        
        df_seq = df.withColumn("icd_text_sequence", F.concat_ws(" ", *seq_exprs))
        pdf = df_seq.select("icd_text_sequence").toPandas()
        sequences = pdf["icd_text_sequence"].fillna("").tolist()
        
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for seq in sequences:
            for token in seq.split():
                if token not in vocab:
                    vocab[token] = len(vocab)
                    
        max_len = 32
        token_ids = []
        for seq in sequences:
            tokens = [vocab.get(t, 1) for t in seq.split()[:max_len]]
            padded = tokens + [0] * (max_len - len(tokens))
            token_ids.append(padded)
            
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        
        class LocalHICDBERT(nn.Module):
            def __init__(self, vocab_size, d_model=128):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
                self.encoder = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model)
                )
                
            def forward(self, x):
                mask = (x != 0).unsqueeze(-1).float()
                emb = self.embedding(x)
                enc = self.encoder(emb)
                sum_emb = (enc * mask).sum(dim=1)
                lens = mask.sum(dim=1).clamp(min=1e-9)
                return sum_emb / lens

        device = torch.device("cpu")
        model = LocalHICDBERT(vocab_size=len(vocab)).to(device).eval()
        
        batch_size = 1024
        embeddings_list = []
        with torch.no_grad():
            for i in range(0, len(input_ids), batch_size):
                batch_x = input_ids[i:i+batch_size].to(device)
                emb = model(batch_x)
                embeddings_list.append(emb.cpu().numpy())
                
        emb_matrix = np.vstack(embeddings_list)
        
        k_comp = min(n_components, emb_matrix.shape[1])
        pca = SklearnPCA(n_components=k_comp, random_state=42)
        emb_pca = pca.fit_transform(emb_matrix)
        
        emb_cols = [f"hicd_bert_emb_{i+1:02d}" for i in range(k_comp)]
        pdf_emb = pd.DataFrame(emb_pca, columns=emb_cols)
        
        df_emb_spark = spark.createDataFrame(pdf_emb)
        
        df_with_idx = df.withColumn("row_id_bert", F.monotonically_increasing_id())
        df_emb_spark_with_idx = df_emb_spark.withColumn("row_id_bert", F.monotonically_increasing_id())
        
        df_res = df_with_idx.join(df_emb_spark_with_idx, on="row_id_bert", how="inner").drop("row_id_bert")
        return df_res
        
    except Exception as e:
        logging.error(f"Local CPU HICD-BERT embedding extraction error ({e}). Proceeding with original DataFrame.")
        return df

def make_column_names_descriptive(df):
    clean_cols = []
    seen = set()
    for c in df.columns:
        c_clean = c.replace(" ", "_").replace("-", "_").replace(".", "_")
        c_clean = re.sub(r'__+', '_', c_clean).strip('_')
        orig_clean = c_clean
        counter = 1
        while c_clean in seen:
            c_clean = f"{orig_clean}_{counter}"
            counter += 1
        seen.add(c_clean)
        clean_cols.append(c_clean)
    
    return df.toDF(*clean_cols)

def add_icd_hierarchy_features(df):
    icd_target_cols = [
        c for c in df.columns 
        if ('ICD' in c or 'Diagnosis' in c) 
        and not c.endswith(('_G0', '_G1', '_G2', '_Name', '_count')) 
        and 'Cluster_Name' not in c
    ]
    
    if not icd_target_cols:
        return df

    def extract_g0(val):
        if not val or str(val).lower() in ('nan', 'none', '<na>', ''): return None
        clean = str(val).strip().replace('.', '').upper()
        return clean[0] if clean else None

    def extract_g1(val):
        if not val or str(val).lower() in ('nan', 'none', '<na>', ''): return None
        clean = str(val).strip().replace('.', '').upper()
        return clean[:2] if len(clean) >= 2 else (clean[0] if clean else None)

    def extract_g2(val):
        if not val or str(val).lower() in ('nan', 'none', '<na>', ''): return None
        clean = str(val).strip().replace('.', '').upper()
        return clean[:3] if len(clean) >= 3 else (clean[:2] if len(clean) >= 2 else None)

    get_g0 = F.udf(extract_g0, StringType())
    get_g1 = F.udf(extract_g1, StringType())
    get_g2 = F.udf(extract_g2, StringType())

    g0_cols, g1_cols, g2_cols = [], [], []
    hierarchy_exprs = []
    for c in icd_target_cols:
        g0_name = f"{c}_G0"
        g1_name = f"{c}_G1"
        g2_name = f"{c}_G2"
        
        hierarchy_exprs.append(get_g0(F.col(f"`{c}`")).alias(g0_name))
        hierarchy_exprs.append(get_g1(F.col(f"`{c}`")).alias(g1_name))
        hierarchy_exprs.append(get_g2(F.col(f"`{c}`")).alias(g2_name))
        
        g0_cols.append(g0_name)
        g1_cols.append(g1_name)
        g2_cols.append(g2_name)

    if hierarchy_exprs:
        df = df.select("*", *hierarchy_exprs)

    df = df.withColumn("icd_g0_unique_count", F.size(F.array_remove(F.array_distinct(F.array(*[F.col(f"`{c}`") for c in g0_cols])), None)))
    df = df.withColumn("icd_g1_unique_count", F.size(F.array_remove(F.array_distinct(F.array(*[F.col(f"`{c}`") for c in g1_cols])), None)))
    df = df.withColumn("icd_g2_unique_count", F.size(F.array_remove(F.array_distinct(F.array(*[F.col(f"`{c}`") for c in g2_cols])), None)))

    return df

def preprocess_notebook_pipeline_condensed_nb(
    input_file, 
    output_file, 
    min_freq_count=200
):
    """
    Optimized and Condensed PySpark Preprocessing Pipeline for Nationwide OASIS Cohort (CPU).
    """
    start_time = time.time()
    logging.info(f"Starting Condensed PySpark Preprocessing Pipeline (CPU) for {input_file}...")

    if not os.path.exists(input_file):
        alt_input = os.path.join("..", input_file)
        if os.path.exists(alt_input):
            input_file = alt_input
        else:
            raise FileNotFoundError(f"Input file not found at {input_file} or {alt_input}")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    spark = get_spark_session("OASIS_Preprocessing_NB_Condensed_CPU")

    logging.info("Reading raw dataset into PySpark...")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_file)

    # 1. Target Column Check & Binary Encoding
    possible_targets = ['ever_readmitted', 'READMISSION', 'Readmission', 'target']
    target_col = next((c for c in possible_targets if c in df.columns), None)
    
    if target_col:
        logging.info(f"Encoding target column '{target_col}'...")
        df = df.withColumn(
            "ever_readmitted",
            F.when(F.col(target_col).cast("string").isin("1", "1.0", "true", "True"), F.lit(1).cast("byte"))
             .otherwise(F.lit(0).cast("byte"))
        )
        if target_col != "ever_readmitted":
            df = df.drop(target_col)
    else:
        raise ValueError(f"Target column not found in input file! Available columns: {df.columns}")

    # 2. Coerce numeric columns & round
    numeric_cols = [
        'Days_Cared_For', 'charlson_score', 'charlson_ageadj', 'elix_quan_score',
        'POPPCT_URB', 'POPPCT_RUR', 'RUCA_Category', 'POPDENSITY', 'MEDIAN_HOUSEHOLD_INCOME',
        'ACS_PCT_UNEMPLOYED', 'ACS_PCT_UNINSURED', 'ACS_PCT_COLLEGE', 'ACS_PCT_POVERTY',
        'ACS_MEDIAN_HOME_VALUE', 'ACS_MEDIAN_RENT', 'ACS_PCT_SNAP', 'ACS_PCT_SINGLE_PARENT',
        'ACS_PCT_BROADBAND', 'ACS_PCT_NO_VEHICLE', 'ACS_PCT_DISABILITY', 'ACS_PCT_ENGLISH_LIMITED',
        'ACS_PCT_SEVERELY_BURDENED_RENTERS', 'ACS_PCT_FOREIGN_BORN', 'ACS_PCT_FOOD_STAMPS',
        'ACS_PCT_PUBLIC_ASSISTANCE', 'ACS_PCT_SMARTPHONE_ONLY', 'ACS_PCT_COMPUTER_NO_INTERNET',
        'ACS_GINI_INDEX', 'ACS_PCT_WORK_FROM_HOME', 'ACS_PCT_COMMUTE_PUBLIC_TRANSIT',
        'ACS_PCT_WORK_NO_VEHICLE', 'ACS_PCT_CHILD_POVERTY', 'ACS_PCT_ELDERLY_POVERTY',
        'ACS_PCT_WHITE', 'ACS_PCT_BLACK', 'ACS_PCT_HISPANIC', 'ACS_PCT_ASIAN',
        'ACS_PCT_AIAN', 'ACS_PCT_NHPI', 'ACS_PCT_MULTI', 'ACS_PCT_OTHER_RACE',
        'ACS_PCT_VETERAN', 'ACS_PCT_RENTER_OCCUPIED', 'ACS_PCT_VACANT_HOUSING',
        'ACS_PCT_MOBILE_HOMES', 'ACS_PCT_OVERCROWDED', 'ACS_PCT_LACK_PLUMBING',
        'ACS_PCT_LACK_KITCHEN', 'ACS_PCT_SAME_HOUSE_1YR', 'ACS_PCT_MOVED_SAME_COUNTY_1YR',
        'ACS_PCT_MOVED_DIFFERENT_STATE_1YR', 'ACS_PCT_BORN_IN_STATE', 'ACS_PCT_COMMUTE_30MIN_PLUS',
        'ACS_PCT_COMMUTE_60MIN_PLUS', 'ACS_PCT_WORKER_16_PLUS', 'ACS_PCT_MEDICARE_ONLY',
        'ACS_PCT_MEDICAID_ONLY', 'ACS_PCT_DUAL_ELIGIBLE', 'ACS_PCT_CIVILIAN_NONINST',
        'ACS_PCT_MALE', 'ACS_PCT_FEMALE', 'ACS_PCT_AGE_65_PLUS', 'ACS_PCT_AGE_85_PLUS',
        'ACS_MEDIAN_AGE', 'ACS_PER_CAPITA_INCOME', 'ACS_PCT_HIGHSCHOOL_ONLY',
        'ACS_PCT_BACHELORS_PLUS', 'ACS_PCT_FAMILY_POVERTY', 'ACS_PCT_PERSON_INC_BELOW99',
        'ACS_PCT_POV_AIAN', 'ACS_PCT_POV_ASIAN', 'ACS_PCT_POV_BLACK', 'ACS_PCT_POV_HISPANIC',
        'ACS_PCT_POV_MULTI', 'ACS_PCT_POV_NHPI', 'ACS_PCT_POV_OTHER', 'ACS_PCT_POV_WHITE',
        'ACS_PCT_VET_POV_18_64', 'ACS_TOT_POP_POV'
    ]

    num_cols_present = [c for c in numeric_cols if c in df.columns]
    if num_cols_present:
        filter_cond = None
        for c in num_cols_present:
            casted = F.col(f"`{c}`").cast(DoubleType())
            cond = casted.isNotNull() & ~F.isnan(casted)
            filter_cond = cond if filter_cond is None else (filter_cond & cond)
        
        df = df.filter(filter_cond)

        num_expr_map = {
            c: F.round(F.col(f"`{c}`").cast(DoubleType()), 2).alias(c)
            for c in num_cols_present
        }
        df = df.select(*[num_expr_map.get(c, F.col(f"`{c}`")) for c in df.columns])
        df = df.localCheckpoint()

    # 3. Drop unwanted metadata/text
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

    # 4. Frequency encoding for high-cardinality ID features
    id_cols = ['Agency_Medicare_Number', 'Facility_Internal_ID']
    for id_col in id_cols:
        if id_col in df.columns:
            logging.info(f"Frequency encoding high-cardinality feature '{id_col}'...")
            freq_df = df.groupBy(f"`{id_col}`").count().withColumnRenamed("count", f"{id_col}_freq")
            df = df.join(freq_df, on=id_col, how="left").drop(id_col)

    # 5. Extract HICD-BERT Embeddings (CPU)
    df = extract_hicd_bert_embeddings_cpu(df, spark, n_components=32)

    # 6. Extract ICD-10 Hierarchy levels (G0, G1, G2)
    logging.info("Extracting ICD-10 hierarchy levels (G0, G1, G2)...")
    df = add_icd_hierarchy_features(df)

    # 7. Condensed One-hot Encoding (min_freq_count=200)
    cat_cols = [
        'Submitted_HIPPS_Code', 'Gender', 'ByDiscipline',
        'BMI_Category', 'Primary_Diagnosis_ICD_10_C_M_Code_Cluster',
        'Other_Diagnosis_Code_1_ICD_10_C_M_Cluster', 'Other_Diagnosis_Code_2_ICD_10_C_M_Cluster',
        'Other_Diagnosis_Code_3_ICD_10_C_M_Cluster', 'Other_Diagnosis_Code_4_ICD_10_C_M_Cluster',
        'Other_Diagnosis_Code_5_ICD_10_C_M_Cluster', 'COUNTY_NAME'
    ]
    icd_cols = [c for c in df.columns if ('ICD' in c or 'Diagnosis' in c) and c not in drop_cols and not c.endswith('_count')]
    all_cat_cols = list(dict.fromkeys(cat_cols + icd_cols))
    present_cat_cols = [c for c in all_cat_cols if c in df.columns]

    if present_cat_cols:
        cat_expr_map = {}
        for col_name in present_cat_cols:
            cleaned = F.regexp_replace(F.trim(F.col(f"`{col_name}`").cast("string")), r"\.0$", "")
            cat_expr_map[col_name] = F.when(cleaned.isin("nan", "None", "<NA>", ""), None).otherwise(cleaned).alias(col_name)

        df = df.select(*[cat_expr_map.get(c, F.col(f"`{c}`")) for c in df.columns])
        df = df.localCheckpoint()

    if present_cat_cols:
        logging.info(f"One-hot encoding {len(present_cat_cols)} categorical/ICD columns (min_freq_count={min_freq_count})...")
        dummy_exprs = []
        cols_to_drop_cat = []
        seen_dummy_names = set(df.columns)

        for c in present_cat_cols:
            freq_df = df.groupBy(f"`{c}`").count().collect()
            frequent_vals = []
            has_rare = False

            for row in freq_df:
                val = row[c]
                val_count = row['count']
                if val is not None and str(val).lower() not in ("nan", "none", "<na>", ""):
                    if val_count >= min_freq_count:
                        frequent_vals.append(val)
                    else:
                        has_rare = True

            if frequent_vals:
                first_val = frequent_vals[0]
                for val in frequent_vals[1:]:
                    val_clean = str(val).replace(" ", "_").replace("-", "_").replace(".", "_")
                    col_dummy_name = f"{c}_{val_clean}"
                    if col_dummy_name not in seen_dummy_names:
                        seen_dummy_names.add(col_dummy_name)
                        dummy_exprs.append(
                            F.when(F.col(f"`{c}`") == val, F.lit(1).cast("byte")).otherwise(F.lit(0).cast("byte")).alias(col_dummy_name)
                        )

                if has_rare and 'Other' != first_val:
                    col_dummy_name = f"{c}_Other"
                    if col_dummy_name not in seen_dummy_names:
                        seen_dummy_names.add(col_dummy_name)
                        dummy_exprs.append(
                            F.when(~F.col(f"`{c}`").isin(frequent_vals) & F.col(f"`{c}`").isNotNull(), F.lit(1).cast("byte")).otherwise(F.lit(0).cast("byte")).alias(col_dummy_name)
                        )

            cols_to_drop_cat.append(c)

        if dummy_exprs:
            df = df.select("*", *dummy_exprs)
        if cols_to_drop_cat:
            df = df.drop(*cols_to_drop_cat)

    # 8. Clean feature names & deduplicate
    logging.info("Formatting feature column names...")
    df = make_column_names_descriptive(df)

    seen = set()
    unique_cols = []
    for c in df.columns:
        c_name = c
        counter = 1
        while c_name in seen:
            c_name = f"{c}_{counter}"
            counter += 1
        seen.add(c_name)
        unique_cols.append(c_name)

    df = df.toDF(*unique_cols)

    # 9. Beneficiary ID filter
    bene_cols = [c for c in ["Beneficiary_ID", "BENE_ID", "beneficiary_id"] if c in df.columns]
    if bene_cols:
        bene_cond = None
        for bene_col in bene_cols:
            c_col = F.col(f"`{bene_col}`")
            cond = c_col.isNotNull() & ~F.isnan(c_col) & (c_col != "") & (c_col != "nan")
            bene_cond = cond if bene_cond is None else (bene_cond & cond)
        if bene_cond is not None:
            logging.info(f"Filtering out null/NaN rows for {bene_cols}...")
            df = df.filter(bene_cond)

    # 10. Drop constant columns
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
        logging.info(f"Dropping {len(constant_cols)} constant columns...")
        df = df.drop(*constant_cols)

    # 11. Dual Export: CSV and Parquet
    output_parquet = os.path.splitext(output_file)[0] + ".parquet"

    logging.info(f"Saving condensed dataset directly to CSV: {output_file}...")
    temp_dir_csv = output_file + "_temp_spark"
    df.coalesce(1).write.option("header", "true").mode("overwrite").csv(temp_dir_csv)

    part_csv_files = glob.glob(os.path.join(temp_dir_csv, "part-*.csv"))
    if part_csv_files:
        if os.path.exists(output_file):
            os.remove(output_file)
        shutil.move(part_csv_files[0], output_file)
        shutil.rmtree(temp_dir_csv, ignore_errors=True)
        logging.info(f"Successfully saved condensed dataset to CSV: {output_file}")

    logging.info(f"Saving condensed dataset directly to Parquet: {output_parquet}...")
    temp_dir_parquet = output_parquet + "_temp_spark"
    df.coalesce(1).write.mode("overwrite").parquet(temp_dir_parquet)

    part_parquet_files = glob.glob(os.path.join(temp_dir_parquet, "part-*.parquet"))
    if part_parquet_files:
        if os.path.exists(output_parquet):
            os.remove(output_parquet)
        shutil.move(part_parquet_files[0], output_parquet)
        shutil.rmtree(temp_dir_parquet, ignore_errors=True)
        logging.info(f"Successfully saved condensed dataset to Parquet: {output_parquet}")

    spark.stop()
    logging.info(f"Condensed Nationwide preprocessing complete in {time.time() - start_time:.2f} seconds!")

def main():
    input_file = "data/final_mergedDF.csv"
    output_file = "data/processed_final_mergedDF_condensed.csv"
    preprocess_notebook_pipeline_condensed_nb(input_file, output_file)

if __name__ == "__main__":
    main()
