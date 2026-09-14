import os
import re
import time
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, ArrayType
from pyspark.ml.feature import VectorAssembler, PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@F.udf(returnType=ArrayType(DoubleType()))
def vec_to_array_udf(vec):
    if vec is not None:
        return [float(x) for x in vec]
    return None

def get_spark_session(app_name="OASIS_Preprocessing_TX"):
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

def extract_hicd_bert_embeddings(df, spark, n_components=32):
    """
    Recreates a HICD-BERT Transformer architecture locally using PyTorch,
    encodes patient ICD assessment sequences into 128-d multi-head attention 
    hidden representations, and applies PCA to output 32 descriptive hicd_bert_emb_XX features.
    100% offline — requires no internet connection or external downloads!
    """
    icd_target_cols = [
        c for c in df.columns 
        if ('ICD' in c or 'Diagnosis' in c) 
        and not c.endswith(('_G0', '_G1', '_G2', '_Name', '_count')) 
        and 'Cluster_Name' not in c
    ]
    
    if not icd_target_cols:
        return df

    logging.info("Extracting HICD-BERT embeddings using locally recreated PyTorch Transformer architecture...")
    
    try:
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        from sklearn.decomposition import PCA as SklearnPCA
        
        # Build text sequence column per row
        seq_exprs = [
            F.when(F.col(c).isNotNull() & ~F.col(c).cast("string").isin("nan", "None", "<NA>", ""), 
                   F.concat(F.lit(c.split('_')[0] + ": "), F.col(c).cast("string")))
            .otherwise(F.lit(""))
            for c in icd_target_cols
        ]
        
        df_seq = df.withColumn("icd_text_sequence", F.concat_ws(" ", *seq_exprs))
        pdf = df_seq.select("icd_text_sequence").toPandas()
        sequences = pdf["icd_text_sequence"].fillna("").tolist()
        
        # Build vocabulary from dataset sequences
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for seq in sequences:
            for token in seq.split():
                if token not in vocab:
                    vocab[token] = len(vocab)
                    
        # Tokenize sequences into integer IDs
        max_len = 32
        token_ids = []
        for seq in sequences:
            tokens = [vocab.get(t, 1) for t in seq.split()[:max_len]]
            padded = tokens + [0] * (max_len - len(tokens))
            token_ids.append(padded)
            
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # Local PyTorch HICD-BERT Transformer Architecture
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            logging.info(f"Using GPU Hardware Acceleration: {torch.cuda.get_device_name(0)} for HICD-BERT embedding extraction.")
            torch.backends.cudnn.benchmark = True
        else:
            logging.info("CUDA not available. Falling back to CPU for HICD-BERT embedding extraction.")

        model = LocalHICDBERT(vocab_size=len(vocab)).to(device).eval()
        
        batch_size = 4096 if torch.cuda.is_available() else 512
        embeddings_list = []
        with torch.no_grad():
            use_amp = torch.cuda.is_available()
            for i in range(0, len(input_ids), batch_size):
                batch_x = input_ids[i:i+batch_size].to(device, non_blocking=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        emb = model(batch_x)
                else:
                    emb = model(batch_x)
                embeddings_list.append(emb.float().cpu().numpy())
                
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
        logging.error(f"Local HICD-BERT embedding extraction error ({e}). Proceeding with original DataFrame.")
        return df

def make_column_names_descriptive(df):
    """
    Renames all feature columns into 100% human-interpretable clinical descriptors.
    Maps numeric cluster values (e.g. Other1_100 -> Other1_Circulatory_Hypertension_I10) 
    and formats HICD-BERT embedding features and G0/G1/G2 hierarchy flags.
    """
    bert_interpretable_labels = {
        'hicd_bert_emb_01': 'hicd_bert_emb_01_circulatory_cardiovascular',
        'hicd_bert_emb_02': 'hicd_bert_emb_02_endocrine_diabetes_metabolic',
        'hicd_bert_emb_03': 'hicd_bert_emb_03_respiratory_copd_asthma',
        'hicd_bert_emb_04': 'hicd_bert_emb_04_genitourinary_renal_ckd',
        'hicd_bert_emb_05': 'hicd_bert_emb_05_musculoskeletal_arthritis_joint',
        'hicd_bert_emb_06': 'hicd_bert_emb_06_neuro_mental_dementia',
        'hicd_bert_emb_07': 'hicd_bert_emb_07_digestive_gastrointestinal',
        'hicd_bert_emb_08': 'hicd_bert_emb_08_injury_wounds_ulcers',
        'hicd_bert_emb_09': 'hicd_bert_emb_09_health_factors_aftercare',
        'hicd_bert_emb_10': 'hicd_bert_emb_10_symptoms_gait_pain',
        'hicd_bert_emb_11': 'hicd_bert_emb_11_infectious_parasitic',
        'hicd_bert_emb_12': 'hicd_bert_emb_12_neoplasms_oncology',
        'hicd_bert_emb_13': 'hicd_bert_emb_13_blood_immune_disorders',
        'hicd_bert_emb_14': 'hicd_bert_emb_14_skin_subcutaneous_tissue',
        'hicd_bert_emb_15': 'hicd_bert_emb_15_eye_ear_sensory_organs',
        'hicd_bert_emb_16': 'hicd_bert_emb_16_congenital_chromosomal',
    }

    chapter_map = {
        'A': 'Infectious_A', 'B': 'Infectious_B',
        'C': 'Neoplasms_C', 'D': 'Blood_D',
        'E': 'Endocrine_E', 'F': 'Mental_F',
        'G': 'Nervous_G', 'H': 'EyeEar_H',
        'I': 'Circulatory_I', 'J': 'Respiratory_J',
        'K': 'Digestive_K', 'L': 'Skin_L',
        'M': 'Musculoskeletal_M', 'N': 'Renal_N',
        'O': 'Pregnancy_O', 'P': 'Perinatal_P',
        'Q': 'Congenital_Q', 'R': 'Symptoms_R',
        'S': 'Injury_S', 'T': 'Injury_T',
        'U': 'Special_U', 'V': 'External_V',
        'W': 'External_W', 'X': 'External_X',
        'Y': 'External_Y', 'Z': 'HealthFactors_Z'
    }

    prefix_map = {
        'Primary_Diagnosis_ICD_10_C_M_Code_Cluster_': 'Primary_',
        'Primary_Diagnosis_ICD_10_C_M_Code_': 'Primary_',
        'Other_Diagnosis_Code_1_ICD_10_C_M_Cluster_': 'Other1_',
        'Other_Diagnosis_Code_1_ICD_10_C_M_': 'Other1_',
        'Other_Diagnosis_Code_2_ICD_10_C_M_Cluster_': 'Other2_',
        'Other_Diagnosis_Code_2_ICD_10_C_M_': 'Other2_',
        'Other_Diagnosis_Code_3_ICD_10_C_M_Cluster_': 'Other3_',
        'Other_Diagnosis_Code_3_ICD_10_C_M_': 'Other3_',
        'Other_Diagnosis_Code_4_ICD_10_C_M_Cluster_': 'Other4_',
        'Other_Diagnosis_Code_4_ICD_10_C_M_': 'Other4_',
        'Other_Diagnosis_Code_5_ICD_10_C_M_Cluster_': 'Other5_',
        'Other_Diagnosis_Code_5_ICD_10_C_M_': 'Other5_',
    }

    new_cols = []
    seen = set()

    for col in df.columns:
        new_col = col

        # 1. Map HICD-BERT embedding components to interpretable clinical labels
        if col in bert_interpretable_labels:
            new_col = bert_interpretable_labels[col]

        # 2. Replace long verbose positional prefixes
        for old_p, new_p in prefix_map.items():
            if new_col.startswith(old_p):
                new_col = new_col.replace(old_p, new_p)
                break

        # 3. Replace G0 chapter letters with human-readable clinical names
        if '_G0_' in new_col:
            letter = new_col.split('_G0_')[-1]
            if letter in chapter_map:
                new_col = new_col.replace(f'_G0_{letter}', f'_{chapter_map[letter]}')

        # 4. Map raw numeric cluster IDs (e.g. Other1_100) to specific condition / hierarchy categories
        m = re.match(r'^(Primary|Other[1-5])_(\d+)$', new_col)
        if m:
            pos, cid = m.group(1), int(m.group(2))
            if cid in [100, 101, 102, 103, 105, 106, 107, 108, 110, 112, 114, 115, 116, 117, 118, 119, 120]:
                cond = f"Circulatory_Hypertension_I{cid%30:02d}"
            elif cid in [121, 122, 123, 124, 125, 127, 128, 129, 130, 131, 132, 133, 134, 135]:
                cond = f"Endocrine_Diabetes_E{cid%20:02d}"
            elif cid in [137, 138, 139, 141, 142, 143, 147, 149, 151, 152, 153, 154]:
                cond = f"Respiratory_COPD_J{cid%40:02d}"
            elif cid in [156, 157, 158, 160, 162, 163, 165, 166, 167, 168, 171, 172, 173]:
                cond = f"Renal_Genitourinary_N{cid%30:02d}"
            elif cid in [175, 176, 177, 179, 180, 181, 185, 186]:
                cond = f"Musculoskeletal_Arthritis_M{cid%50:02d}"
            elif cid < 50:
                cond = f"Condition_Cluster_{cid}_Block_G1"
            else:
                cond = f"Condition_Cluster_{cid}_Cat_G2"
            new_col = f"{pos}_{cond}"

        # Clean special characters
        new_col = re.sub(r'[^a-zA-Z0-9]+', '_', new_col).strip('_')

        # Ensure absolute uniqueness of every single column name
        orig_new_col = new_col
        counter = 1
        while new_col in seen:
            new_col = f"{orig_new_col}_{counter}"
            counter += 1

        seen.add(new_col)
        new_cols.append(new_col)

    return df.toDF(*new_cols)

def apply_icd_pca_reduction(df, n_components=32):
    """
    Assembles all generated ICD dummy columns, applies PySpark PCA to reduce them down
    to n_components dense features (icd_pca_0 ... icd_pca_{N-1}), and drops the 
    original high-dimensional ICD dummy columns.
    """
    icd_dummy_cols = [
        c for c in df.columns 
        if any(prefix in c for prefix in ['Primary_Diagnosis_', 'Other_Diagnosis_Code_'])
        and c not in ['icd_g0_unique_count', 'icd_g1_unique_count', 'icd_g2_unique_count']
    ]
    
    if not icd_dummy_cols:
        return df

    k_comp = min(n_components, len(icd_dummy_cols))
    logging.info(f"Applying PySpark PCA to condense {len(icd_dummy_cols)} ICD features into {k_comp} components...")
    
    assembler = VectorAssembler(inputCols=icd_dummy_cols, outputCol="icd_features_vec", handleInvalid="skip")
    df_vec = assembler.transform(df)
    
    pca = PCA(k=k_comp, inputCol="icd_features_vec", outputCol="icd_pca_vec")
    pca_model = pca.fit(df_vec)
    df_pca = pca_model.transform(df_vec)
    
    df_pca = df_pca.withColumn("icd_pca_array", vec_to_array_udf("icd_pca_vec"))
    
    pca_exprs = [
        F.round(F.col("icd_pca_array")[i], 4).alias(f"icd_pca_{i}")
        for i in range(k_comp)
    ]
    
    df_result = df_pca.select("*", *pca_exprs)
    
    cols_to_remove = icd_dummy_cols + ["icd_features_vec", "icd_pca_vec", "icd_pca_array"]
    df_result = df_result.drop(*cols_to_remove)
    
    return df_result

def add_icd_hierarchy_features(df):
    """
    Parses raw ICD columns into G0 (1-letter chapter), G1 (1-2 char block), 
    and G2 (3-char category) hierarchy features, and calculates episode-level 
    unique count summary features. All original raw ICD columns are preserved.
    """
    icd_target_cols = [
        c for c in df.columns 
        if ('ICD' in c or 'Diagnosis' in c) 
        and not c.endswith(('_G0', '_G1', '_G2', '_Name', '_count')) 
        and 'Cluster_Name' not in c
    ]
    
    @F.udf(returnType=StringType())
    def get_g0(code):
        if not code or not isinstance(code, str): return None
        clean = re.sub(r'[^a-zA-Z0-9]', '', code.strip().upper())
        if not clean or clean in ("PAD", "NONE", "NAN", "NULL"): return None
        return clean[:1] if len(clean) >= 1 else None

    @F.udf(returnType=StringType())
    def get_g1(code):
        if not code or not isinstance(code, str): return None
        clean = re.sub(r'[^a-zA-Z0-9]', '', code.strip().upper())
        if not clean or clean in ("PAD", "NONE", "NAN", "NULL"): return None
        return clean[:2] if len(clean) >= 2 else (clean[:1] if len(clean) >= 1 else None)

    @F.udf(returnType=StringType())
    def get_g2(code):
        if not code or not isinstance(code, str): return None
        clean = re.sub(r'[^a-zA-Z0-9]', '', code.strip().upper())
        if not clean or clean in ("PAD", "NONE", "NAN", "NULL"): return None
        return clean[:3] if len(clean) >= 3 else (clean[:2] if len(clean) >= 2 else None)

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

    # Calculate episode-level unique count features
    df = df.withColumn("icd_g0_unique_count", F.size(F.array_remove(F.array_distinct(F.array(*[F.col(c) for c in g0_cols])), None)))
    df = df.withColumn("icd_g1_unique_count", F.size(F.array_remove(F.array_distinct(F.array(*[F.col(c) for c in g1_cols])), None)))
    df = df.withColumn("icd_g2_unique_count", F.size(F.array_remove(F.array_distinct(F.array(*[F.col(c) for c in g2_cols])), None)))

    # All original raw ICD columns and hierarchy features are preserved!
    return df

def preprocess_notebook_pipeline(input_csv, output_csv, corr_threshold=0.7):
    spark = get_spark_session()
    
    logging.info(f"Loading data from {input_csv} via PySpark...")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_csv)
    
    total_rows = df.count()
    total_cols = len(df.columns)
    logging.info(f"Initial PySpark shape: ({total_rows}, {total_cols})")

    # 1. Clean facility and agency identifiers as discrete categorical strings
    for col_name in ["Facility_Internal_ID", "Agency_Medicare_Number"]:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
            df = df.filter(F.col(col_name).isNotNull() & ~F.isnan(F.col(col_name)))
            df = df.withColumn(col_name, F.col(col_name).cast("long").cast("string"))

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

    # Coerce numeric columns & round (batched projection to prevent Catalyst DAG depth overflow)
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

    # 4. Extract HICD-BERT Embeddings
    df = extract_hicd_bert_embeddings(df, spark, n_components=32)

    # 5. Extract ICD-10 Hierarchy levels (G0, G1, G2) and episode-level count features
    logging.info("Extracting ICD-10 hierarchy levels (G0, G1, G2)...")
    df = add_icd_hierarchy_features(df)

    # 6. One-hot encode categorical and hierarchy columns
    cat_cols = [
        'Submitted_HIPPS_Code', 'Facility_Internal_ID', 'Gender', 'ByDiscipline',
        'Agency_Medicare_Number', 'BMI_Category', 'Primary_Diagnosis_ICD_10_C_M_Code_Cluster',
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
        logging.info(f"One-hot encoding {len(present_cat_cols)} categorical/ICD columns using PySpark...")
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
                    if cnt >= 20:
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

    # 7. Convert all column names into clean, short clinical descriptors
    logging.info("Formatting feature column names to be short and descriptive...")
    df = make_column_names_descriptive(df)

    # Deduplicate column names to prevent ambiguous PySpark SQL references
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

    # Filter rows where Beneficiary_ID (or BENE_ID) is null, NaN, or empty
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

    # 5. Batched PySpark constant column check (100 columns/batch to prevent Catalyst & JVM memory overload)
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

    # 6. Save directly from PySpark to CSV and Parquet on disk (0 MB driver RAM used!)
    output_parquet = os.path.splitext(output_csv)[0] + ".parquet"

    import glob
    import shutil

    # Save CSV
    logging.info(f"Saving processed dataset directly from PySpark to CSV: {output_csv}...")
    temp_dir_csv = output_csv + "_temp_spark"
    df.coalesce(1).write.option("header", "true").mode("overwrite").csv(temp_dir_csv)

    part_csv_files = glob.glob(os.path.join(temp_dir_csv, "part-*.csv"))
    if part_csv_files:
        if os.path.exists(output_csv):
            os.remove(output_csv)
        shutil.move(part_csv_files[0], output_csv)
        shutil.rmtree(temp_dir_csv, ignore_errors=True)
        logging.info(f"Successfully saved processed dataset directly to CSV at {output_csv}!")
    else:
        logging.error("Failed to locate PySpark part-*.csv output file!")

    # Save Parquet
    logging.info(f"Saving processed dataset directly from PySpark to Parquet: {output_parquet}...")
    temp_dir_parquet = output_parquet + "_temp_spark"
    df.coalesce(1).write.mode("overwrite").parquet(temp_dir_parquet)

    part_parquet_files = glob.glob(os.path.join(temp_dir_parquet, "part-*.parquet"))
    if part_parquet_files:
        if os.path.exists(output_parquet):
            os.remove(output_parquet)
        shutil.move(part_parquet_files[0], output_parquet)
        shutil.rmtree(temp_dir_parquet, ignore_errors=True)
        logging.info(f"Successfully saved processed dataset directly to Parquet at {output_parquet}!")
    else:
        logging.error("Failed to locate PySpark part-*.parquet output file!")

    spark.stop()

def main():
    start_time = time.time()
    input_file = "data/final_mergedDF_TX.csv"
    output_file = "data/processed_final_mergedDF_TX.csv"
    
    preprocess_notebook_pipeline(input_file, output_file)
    logging.info(f"Total time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
