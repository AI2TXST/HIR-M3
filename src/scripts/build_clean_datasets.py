import os
import re
import gc
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SimpleHICDBertEncoder(nn.Module):
    """
    Lightweight PyTorch Multi-Head Self-Attention model to encode ICD sequence tokens.
    """
    def __init__(self, vocab_size=2000, emb_dim=64, n_heads=4, hidden_dim=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(20, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=hidden_dim, 
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.proj = nn.Linear(emb_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        emb = self.token_emb(x) + self.pos_emb(pos)
        out = self.transformer(emb)
        pooled = out.mean(dim=1)
        return self.proj(pooled)

def extract_hicd_bert_features(df, icd_cols, n_components=32, seed=42):
    logging.info("Extracting HICD-BERT multi-head attention sequence representations...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build sequence of ICD tokens
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for c in icd_cols:
        for val in df[c].dropna().unique():
            val_str = str(val).strip()
            if val_str and val_str not in vocab:
                vocab[val_str] = len(vocab)

    vocab_size = min(len(vocab) + 10, 5000)
    token_matrix = np.zeros((len(df), len(icd_cols)), dtype=np.int64)

    for col_idx, c in enumerate(icd_cols):
        mapped = df[c].astype(str).map(vocab).fillna(1).values
        token_matrix[:, col_idx] = mapped

    model = SimpleHICDBertEncoder(vocab_size=vocab_size, emb_dim=64, n_heads=4, hidden_dim=64)
    model.eval()

    embeddings = []
    batch_size = 8192
    with torch.no_grad():
        for i in range(0, len(token_matrix), batch_size):
            batch_tokens = torch.tensor(token_matrix[i:i+batch_size], dtype=torch.long)
            batch_emb = model(batch_tokens).cpu().numpy()
            embeddings.append(batch_emb)

    full_embeddings = np.vstack(embeddings)

    # PCA down to n_components
    pca = PCA(n_components=n_components, random_state=seed)
    pca_emb = pca.fit_transform(full_embeddings).astype(np.float32)

    emb_df = pd.DataFrame(
        pca_emb,
        columns=[f"hicd_bert_emb_{i}" for i in range(n_components)],
        index=df.index
    )
    logging.info(f"HICD-BERT embedding extracted: shape {emb_df.shape}")
    return emb_df

def build_leak_free_datasets():
    start_time = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)

    logging.info("Step 1: Loading raw OASIS clustered ICD dataset...")
    df_oasis = pd.read_csv("data/oasis_clusteredICD.csv", engine="python", on_bad_lines="skip")
    logging.info(f"Loaded {len(df_oasis):,} rows, {len(df_oasis.columns)} columns.")

    # Target definition
    target_col = "ever_readmitted"
    df_oasis[target_col] = pd.to_numeric(df_oasis[target_col], errors="coerce").fillna(0).astype(np.int8)

    # Clean zip and state for merging
    if "zip" not in df_oasis.columns and "Patient_ZIP_Code" in df_oasis.columns:
        df_oasis["zip"] = df_oasis["Patient_ZIP_Code"].astype(str).str.strip().str[:5]
    elif "zip" in df_oasis.columns:
        df_oasis["zip"] = df_oasis["zip"].astype(str).str.strip().str[:5]

    if "state_id" not in df_oasis.columns and "STATE_ID" in df_oasis.columns:
        df_oasis["state_id"] = df_oasis["STATE_ID"].astype(str).str.strip().str.upper()
    elif "state_id" in df_oasis.columns:
        df_oasis["state_id"] = df_oasis["state_id"].astype(str).str.strip().str.upper()

    logging.info("Step 2: Loading and preparing SDOH Census / ZIP data...")
    sdoh_tx_path = "data/sdoh_zips_census_TX.csv"
    if os.path.exists(sdoh_tx_path):
        sdoh_df = pd.read_csv(sdoh_tx_path, low_memory=False)
        sdoh_df["zip"] = sdoh_df["zip"].astype(str).str.strip().str.zfill(5).str[:5]
        sdoh_df["state_id"] = sdoh_df["state_id"].astype(str).str.strip().str.upper()
        sdoh_df = sdoh_df.drop_duplicates(subset=["state_id", "zip"])
    else:
        sdoh_df = None

    # Load uszips for county names and urban/rural if needed
    if os.path.exists("data/uszips.csv"):
        uszips = pd.read_csv("data/uszips.csv", usecols=["zip", "county_name", "county_fips", "state_id"])
        uszips["zip"] = uszips["zip"].astype(str).str.strip().str.zfill(5).str[:5]
        uszips["state_id"] = uszips["state_id"].astype(str).str.strip().str.upper()
        uszips = uszips.drop_duplicates(subset=["state_id", "zip"])
    else:
        uszips = None

    logging.info("Step 3: Merging OASIS with SDOH / Geographic indicators...")
    if sdoh_df is not None:
        merged = df_oasis.merge(sdoh_df, on=["state_id", "zip"], how="left")
    elif uszips is not None:
        merged = df_oasis.merge(uszips, on=["state_id", "zip"], how="left")
    else:
        merged = df_oasis.copy()

    # Fill any missing SDOH columns using national averages from DF_summary if present
    if os.path.exists("data/DF_summary.csv"):
        df_sum = pd.read_csv("data/DF_summary.csv")
        num_sdoh = [c for c in df_sum["column"] if (c.startswith("ACS_") or c.startswith("POP")) and c in merged.columns]
        for c in num_sdoh:
            if merged[c].isnull().any():
                merged[c] = merged[c].fillna(merged[c].median())

    logging.info(f"Merged dataset shape: {merged.shape}")

    # ENFORCE STRICT AUDIT: Drop lookahead leakage & post-index duration variables
    lookahead_leakage_drops = [
        "Days_Cared_For", "ever_deceased", "NumVisits", "DaysBetweenVisits",
        "PrevVisitDate", "Last_Assessment_Date", "Assessment_Effective_Date",
        "Patient_ZIP_Code", "STATE_ID", "zip", "city", "state_name", "COUNTYFIPS",
        "county_fips", "blane", "diab", "diabc", "diabunc", "diabwc", "msld", "pud"
    ]
    to_drop = [c for c in lookahead_leakage_drops if c in merged.columns]
    merged.drop(columns=to_drop, inplace=True, errors="ignore")
    logging.info(f"Dropped {len(to_drop)} lookahead leakage / ID columns: {to_drop}")

    # Extract HICD-BERT Embeddings
    icd_cluster_cols = [c for c in merged.columns if "ICD_10_C_M_Cluster" in c or "Code_Cluster" in c]
    icd_clean_cols = [c for c in icd_cluster_cols if not c.endswith("_Name")]
    
    if icd_clean_cols:
        emb_df = extract_hicd_bert_features(merged, icd_clean_cols, n_components=32)
        merged = pd.concat([merged, emb_df], axis=1)

    # Frequency encode high-cardinality IDs
    for id_col in ["Agency_Medicare_Number", "Facility_Internal_ID"]:
        if id_col in merged.columns:
            freq = merged[id_col].value_counts()
            merged[f"{id_col}_freq"] = merged[id_col].map(freq).astype(np.float32)
            merged.drop(columns=[id_col], inplace=True)

    # One-hot encode categorical features
    cat_cols_to_encode = ["Submitted_HIPPS_Code", "Gender", "ByDiscipline", "BMI_Category", "COUNTY_NAME"]
    cat_present = [c for c in cat_cols_to_encode if c in merged.columns]
    if cat_present:
        logging.info(f"One-hot encoding categoricals: {cat_present}...")
        merged = pd.get_dummies(merged, columns=cat_present, drop_first=True, dtype=np.int8)

    # Drop text columns
    text_drops = [c for c in merged.columns if c.endswith("_Name_In_Row") or c.endswith("_Cluster_Name") or c in ["ICD_Clusters_In_Row", "ICD_Cluster_Names_In_Row", "Primary_Diagnosis_ICD_10_C_M_Code_Cluster_Name"]]
    merged.drop(columns=text_drops, inplace=True, errors="ignore")

    # Clean and standardize column names
    clean_cols = {}
    for c in merged.columns:
        clean = re.sub(r'[,:\"\'\[\]{}()<>=]', '', str(c)).replace(' ', '_').replace('-', '_')
        clean_cols[c] = clean
    merged.rename(columns=clean_cols, inplace=True)

    # Split into Texas and Nationwide cohorts
    is_tx = merged["state_id"].str.upper() == "TX" if "state_id" in merged.columns else pd.Series(False, index=merged.index)
    df_tx = merged[is_tx].copy().drop(columns=["state_id"], errors="ignore")
    df_nationwide = merged.copy().drop(columns=["state_id"], errors="ignore")

    # Final downcasting
    for c in df_nationwide.columns:
        if c == target_col:
            continue
        if df_nationwide[c].dtype == "object":
            df_nationwide[c] = pd.to_numeric(df_nationwide[c], errors="coerce").fillna(0.0)
            df_tx[c] = pd.to_numeric(df_tx[c], errors="coerce").fillna(0.0)

    # Save Nationwide Dataset
    out_nat_csv = "data/processed_final_mergedDF_condensed.csv"
    logging.info(f"Saving Nationwide dataset: {out_nat_csv} ({df_nationwide.shape})...")
    df_nationwide.to_csv(out_nat_csv, index=False)

    # Save Texas Dataset
    out_tx_csv = "data/processed_final_mergedDF_condensed_TX.csv"
    logging.info(f"Saving Texas dataset: {out_tx_csv} ({df_tx.shape})...")
    df_tx.to_csv(out_tx_csv, index=False)

    logging.info(f"Dataset build complete in {time.time() - start_time:.2f} seconds!")
    logging.info(f"Verified: 'Days_Cared_For' in Nationwide? {'Days_Cared_For' in df_nationwide.columns}")
    logging.info(f"Verified: 'Days_Cared_For' in Texas? {'Days_Cared_For' in df_tx.columns}")

if __name__ == "__main__":
    build_leak_free_datasets()
