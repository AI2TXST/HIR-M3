import os
import sys
import gc
import pandas as pd
import numpy as np
from utils import (
    load_and_preprocess_data, log_time_and_step, 
    AVAILABLE_FS_METHODS, LGBM_AVAILABLE, XGB_AVAILABLE, IMBLEARN_AVAILABLE
)

TARGET = "ever_readmitted"

DATASETS_TO_RUN = {
    "Nationwide_condensed": [
        "../data/processed_final_mergedDF_condensed.csv",
        "data/processed_final_mergedDF_condensed.csv",
        "../data/processed_final_mergedDF.csv",
        "data/processed_final_mergedDF.csv"
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
    if "has_diabetes" in df_model_reduced.columns:
        df_diabetes = df_model_reduced[df_model_reduced["has_diabetes"] == 1].copy()
        df_diabetes = df_diabetes.drop(columns=["has_diabetes"], errors='ignore')
        yield "Diabetic Patients", df_diabetes
    elif any(c in df_model_reduced.columns for c in ["diab", "diabc", "diabunc"]):
        diab_col = next(c for c in ["diab", "diabc", "diabunc"] if c in df_model_reduced.columns)
        df_diabetes = df_model_reduced[df_model_reduced[diab_col] == 1].copy()
        yield "Diabetic Patients", df_diabetes

    # 3. Cohort: Heart Failure Patients
    if "has_heart_failure" in df_model_reduced.columns:
        df_hf = df_model_reduced[df_model_reduced["has_heart_failure"] == 1].copy()
        df_hf = df_hf.drop(columns=["has_heart_failure"], errors='ignore')
        yield "Heart Failure Patients", df_hf
    elif "chf" in df_model_reduced.columns:
        df_hf = df_model_reduced[df_model_reduced["chf"] == 1].copy()
        yield "Heart Failure Patients", df_hf

    # 4. Cohort: Hypertensive Patients
    if "has_hypertension" in df_model_reduced.columns:
        df_htn = df_model_reduced[df_model_reduced["has_hypertension"] == 1].copy()
        df_htn = df_htn.drop(columns=["has_hypertension"], errors='ignore')
        yield "Hypertensive Patients", df_htn
    elif any(c in df_model_reduced.columns for c in ["hypc", "hypunc", "hp"]):
        htn_col = next(c for c in ["hypc", "hypunc", "hp"] if c in df_model_reduced.columns)
        df_htn = df_model_reduced[df_model_reduced[htn_col] == 1].copy()
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

    parquet_file = base_file.rsplit('.', 1)[0] + '.parquet'
    loaded = False
    if os.path.exists(parquet_file):
        try:
            log_time_and_step(f"   Loading dataset directly from Parquet: {parquet_file}...")
            df_model_reduced = pd.read_parquet(parquet_file)
            loaded = True
        except Exception as e:
            log_time_and_step(f"   Parquet reader unavailable ({e}). Falling back to CSV.")

    if not loaded:
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

    # Downcast float64 to float32
    float_cols = df_model_reduced.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        df_model_reduced[float_cols] = df_model_reduced[float_cols].astype(np.float32)

    gc.collect()

    log_time_and_step(f"Loaded full dataset with shape: {df_model_reduced.shape}")

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
    log_time_and_step("--- Feature Selection Pipeline Started ---")
    log_time_and_step(f"Library Status -> LightGBM: {LGBM_AVAILABLE}, XGBoost: {XGB_AVAILABLE}, Imblearn: {IMBLEARN_AVAILABLE}")

    for key, paths in DATASETS_TO_RUN.items():
        process_feature_selection_for_dataset(key, paths)

    log_time_and_step("--- Feature Selection Pipeline Finished ---")


if __name__ == "__main__":
    main()
