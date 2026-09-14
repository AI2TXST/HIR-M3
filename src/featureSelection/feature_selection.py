import os
import sys
import gc
import pandas as pd
import numpy as np
from utils import (
    load_and_preprocess_data, log_time_and_step, 
    AVAILABLE_FS_METHODS, LGBM_AVAILABLE, XGB_AVAILABLE, IMBLEARN_AVAILABLE
)

import concurrent.futures

TARGET = "ever_readmitted"

DATASETS_TO_RUN = {
    "Texas": [
        "../data/processed_final_mergedDF_condensed_TX.parquet",
        "data/processed_final_mergedDF_condensed_TX.parquet",
        "../data/processed_final_mergedDF_condensed_TX.csv",
        "data/processed_final_mergedDF_condensed_TX.csv"
    ],
    "Nationwide": [
        "../data/processed_final_mergedDF_condensed.parquet",
        "data/processed_final_mergedDF_condensed.parquet",
        "../data/processed_final_mergedDF_condensed.csv",
        "data/processed_final_mergedDF_condensed.csv"
    ]
}

def yield_cohorts(df_model_reduced):
    """
    Yields (cohort_name, cohort_dataframe) one at a time to prevent keeping multiple
    copies of large datasets in RAM simultaneously.
    """
    # 1. Complete Cohort
    yield "Complete Cohort", df_model_reduced.copy()

    # 2. Cohort: Diabetic Patients
    diab_cols = [c for c in df_model_reduced.columns if c in ["has_diabetes", "diab", "diabc", "diabunc"] or "Endocrine_Diabetes" in c]
    if diab_cols:
        diab_col = diab_cols[0]
        df_diabetes = df_model_reduced[df_model_reduced[diab_col] == 1].copy()
        if diab_col == "has_diabetes":
            df_diabetes = df_diabetes.drop(columns=["has_diabetes"], errors='ignore')
        yield "Diabetic Patients", df_diabetes

    # 3. Cohort: Heart Failure Patients
    hf_cols = [c for c in df_model_reduced.columns if c in ["has_heart_failure", "chf", "carit"] or "HeartFailure" in c]
    if hf_cols:
        hf_col = hf_cols[0]
        df_hf = df_model_reduced[df_model_reduced[hf_col] == 1].copy()
        if hf_col == "has_heart_failure":
            df_hf = df_hf.drop(columns=["has_heart_failure"], errors='ignore')
        yield "Heart Failure Patients", df_hf

    # 4. Cohort: Hypertensive Patients
    htn_cols = [c for c in df_model_reduced.columns if c in ["has_hypertension", "hypc", "hypunc", "hp"] or "Hypertension" in c]
    if htn_cols:
        htn_col = htn_cols[0]
        df_htn = df_model_reduced[df_model_reduced[htn_col] == 1].copy()
        if htn_col == "has_hypertension":
            df_htn = df_htn.drop(columns=["has_hypertension"], errors='ignore')
        yield "Hypertensive Patients", df_htn


def process_feature_selection_for_dataset(dataset_key, possible_files):
    base_file = next((f for f in possible_files if os.path.exists(f)), None)
    if not base_file:
        log_time_and_step(f"Warning: Could not find dataset for {dataset_key}. Skipping.")
        return

    log_time_and_step(f"\n=========================================================")
    log_time_and_step(f"   RUNNING FEATURE SELECTION FOR: {dataset_key}")
    log_time_and_step(f"   Loading dataset from {base_file}...")
    log_time_and_step(f"=========================================================")

    if base_file.endswith('.parquet'):
        try:
            log_time_and_step(f"   Loading dataset directly from Parquet: {base_file}...")
            df_model_reduced = pd.read_parquet(base_file)
        except Exception as e:
            log_time_and_step(f"   Parquet reader unavailable ({e}). Falling back to CSV.")
            csv_file = base_file.rsplit('.', 1)[0] + '.csv'
            df_model_reduced = pd.read_csv(csv_file, low_memory=False)
    else:
        log_time_and_step(f"   Loading dataset directly from CSV: {base_file}...")
        df_model_reduced = pd.read_csv(base_file, low_memory=False)

    from sklearn.model_selection import train_test_split

    # Downcast int64 to int8/int16 to save memory
    int_cols = df_model_reduced.select_dtypes(include=['int64', 'int32']).columns
    for c in int_cols:
        c_min, c_max = df_model_reduced[c].min(), df_model_reduced[c].max()
        if c_min >= -128 and c_max <= 127:
            df_model_reduced[c] = df_model_reduced[c].astype(np.int8)
        elif c_min >= -32768 and c_max <= 32767:
            df_model_reduced[c] = df_model_reduced[c].astype(np.int16)
        else:
            df_model_reduced[c] = df_model_reduced[c].astype(np.int32)

    # 50% Stratified Sampling for Nationwide; 100% for Texas
    possible_targets = [TARGET, 'ever_readmitted', 'READMISSION', 'Readmission']
    t_col = next((t for t in possible_targets if t in df_model_reduced.columns), None)
    if "TX" not in dataset_key and "texas" not in dataset_key.lower() and "TX" not in base_file:
        log_time_and_step(f"   Sampling Nationwide dataset ({len(df_model_reduced):,} rows) by 50% to prevent MemoryError...")
        if t_col:
            df_model_reduced, _ = train_test_split(df_model_reduced, train_size=0.50, stratify=df_model_reduced[t_col], random_state=42)
        else:
            df_model_reduced = df_model_reduced.sample(frac=0.50, random_state=42).copy()
    else:
        log_time_and_step(f"   Using 100% of Texas dataset ({len(df_model_reduced):,} rows)...")

    gc.collect()

    log_time_and_step(f"Loaded feature selection dataset shape: {df_model_reduced.shape}")

    all_fs_rows = []

    for name, cohort_df in yield_cohorts(df_model_reduced):
        log_time_and_step(f"Processing Cohort for FS: {name} (shape: {cohort_df.shape})")

        data = load_and_preprocess_data(cohort_df, name, TARGET)
        del cohort_df
        gc.collect()

        if not data: continue
        X_train, X_test, y_train, y_test = data

        feature_names = X_train.columns.tolist()
        fs_res = []

        for m_name, m_func in AVAILABLE_FS_METHODS.items():
            print(f"  Running FS Method: {m_name}")
            try:
                r = m_func(X_train, y_train, X_test, y_test)
                if r:
                    r['method'] = m_name
                    fs_res.append(r)
            except Exception as e:
                print(f"    FS Error {m_name}: {e}")

        cohort_sample_size = len(X_train) + len(X_test)
        num_features_cnt = len(feature_names)

        for result in fs_res:
            method = result['method']
            indices = result['indices']
            scores = result['scores']
            for idx, score in zip(indices, scores):
                if idx < len(feature_names):
                    all_fs_rows.append({
                        'Dataset': name,
                        'Cohort Size': cohort_sample_size,
                        'Num Features': num_features_cnt,
                        'Method': method,
                        'Feature Name': feature_names[idx],
                        'Score': score
                    })

        del X_train, X_test, y_train, y_test
        gc.collect()

    del df_model_reduced
    gc.collect()

    if all_fs_rows:
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{dataset_key}_feature_selection_results.csv")
        log_time_and_step(f"Saving separate FS results ({len(all_fs_rows)} rows) to {out_path}...")
        pd.DataFrame(all_fs_rows).to_csv(out_path, index=False)


def main():
    log_time_and_step("--- Parallel Feature Selection Pipeline Started (Texas & Nationwide) ---")
    log_time_and_step(f"Library Status -> LightGBM: {LGBM_AVAILABLE}, XGBoost: {XGB_AVAILABLE}, Imblearn: {IMBLEARN_AVAILABLE}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(process_feature_selection_for_dataset, key, paths): key
            for key, paths in DATASETS_TO_RUN.items()
        }
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                future.result()
                log_time_and_step(f"Finished feature selection for {key}.")
            except Exception as e:
                log_time_and_step(f"Error in parallel feature selection for {key}: {e}")

    log_time_and_step("--- Feature Selection Pipeline Finished ---")


if __name__ == "__main__":
    main()
