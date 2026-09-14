import os
import sys
import gc
import logging
import pandas as pd
import numpy as np

# Ensure parent and current directory in PYTHONPATH for imports
FS_DIR = os.path.dirname(os.path.abspath(__file__))
if FS_DIR not in sys.path:
    sys.path.insert(0, FS_DIR)

from utils import (
    load_and_preprocess_data, log_time_and_step,
    AVAILABLE_FS_METHODS, LGBM_AVAILABLE, XGB_AVAILABLE, CATBOOST_AVAILABLE, TORCH_AVAILABLE, IMBLEARN_AVAILABLE
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


def yield_urban_rural_cohorts(df_model):
    """
    Yields (subgroup_name, cohort_dataframe) for:
      1. Complete Cohort (All rows)
      2. Urban Subgroup (RUCA / POPPCT_URB >= 50.0)
      3. Rural Subgroup (RUCA / POPPCT_URB < 50.0)
    """
    # 1. Complete Cohort
    yield "Complete Cohort", df_model.copy()

    # Determine Urban vs Rural slicing columns
    urb_col = next((c for c in ['POPPCT_URB', 'POPPCT_URB_condensed', 'urban_pct'] if c in df_model.columns), None)
    ruca_col = next((c for c in ['area_type', 'ruca_category', 'ruca_group', 'urban_rural', 'RUCA_Category'] if c in df_model.columns), None)

    df_urban, df_rural = None, None

    if ruca_col:
        log_time_and_step(f"  Filtering Urban/Rural using RUCA column: '{ruca_col}'")
        urban_mask = df_model[ruca_col].astype(str).str.lower().isin(['urban', '1', '1.0'])
        df_urban = df_model[urban_mask].copy()
        df_rural = df_model[~urban_mask].copy()
    elif urb_col:
        log_time_and_step(f"  Filtering Urban/Rural using POPPCT_URB column: '{urb_col}'")
        urban_mask = df_model[urb_col] >= 50.0
        df_urban = df_model[urban_mask].copy()
        df_rural = df_model[~urban_mask].copy()
    else:
        log_time_and_step("  Warning: Neither RUCA_Category nor POPPCT_URB column found. Skipping Urban/Rural subgroup split.")

    if df_urban is not None and not df_urban.empty:
        yield "Urban Subgroup", df_urban
    
    if df_rural is not None and not df_rural.empty:
        yield "Rural Subgroup", df_rural


def process_fs_for_dataset(dataset_name, possible_paths):
    filepath = next((p for p in possible_paths if os.path.exists(p)), None)
    if not filepath:
        log_time_and_step(f"Warning: Could not find dataset files for '{dataset_name}'. Tried paths: {possible_paths}")
        return []

    log_time_and_step(f"\n=========================================================")
    log_time_and_step(f"   STARTING FEATURE SELECTION: {dataset_name.upper()} DATASET")
    log_time_and_step(f"   Source Path: {filepath}")
    log_time_and_step(f"=========================================================")

    if filepath.endswith('.parquet'):
        try:
            log_time_and_step(f"   Loading directly from Parquet: {filepath}...")
            df = pd.read_parquet(filepath)
        except Exception as e:
            log_time_and_step(f"   Parquet load failed ({e}). Falling back to CSV.")
            csv_path = filepath.rsplit('.', 1)[0] + '.csv'
            df = pd.read_csv(csv_path, low_memory=False)
    else:
        log_time_and_step(f"   Loading directly from CSV: {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)

    # Memory downcasting for integer and float columns
    int_cols = df.select_dtypes(include=['int64', 'int32']).columns
    for c in int_cols:
        c_min, c_max = df[c].min(), df[c].max()
        if c_min >= -128 and c_max <= 127:
            df[c] = df[c].astype(np.int8)
        elif c_min >= -32768 and c_max <= 32767:
            df[c] = df[c].astype(np.int16)
        else:
            df[c] = df[c].astype(np.int32)

    # 50% Stratified Sampling for Nationwide; 100% for Texas
    from sklearn.model_selection import train_test_split
    possible_targets = [TARGET, 'ever_readmitted', 'READMISSION', 'Readmission']
    t_col = next((t for t in possible_targets if t in df.columns), None)
    if "TX" not in dataset_name and "texas" not in dataset_name.lower() and "TX" not in filepath:
        log_time_and_step(f"   Sampling Nationwide dataset ({len(df):,} rows) by 50% to prevent MemoryError...")
        if t_col:
            df, _ = train_test_split(df, train_size=0.50, stratify=df[t_col], random_state=42)
        else:
            df = df.sample(frac=0.50, random_state=42).copy()
    else:
        log_time_and_step(f"   Using 100% of Texas dataset ({len(df):,} rows)...")

    gc.collect()
    log_time_and_step(f"   Loaded dataset shape for {dataset_name}: {df.shape}")

    dataset_fs_rows = []

    for subgroup_name, subgroup_df in yield_urban_rural_cohorts(df):
        log_time_and_step(f"\n--- Processing Subgroup: {dataset_name} ({subgroup_name}) - Shape: {subgroup_df.shape} ---")

        data = load_and_preprocess_data(subgroup_df, f"{dataset_name}_{subgroup_name}", TARGET)
        del subgroup_df
        gc.collect()

        if not data:
            log_time_and_step(f"Warning: Data preprocessing failed for {dataset_name} ({subgroup_name}). Skipping.")
            continue

        X_train, X_test, y_train, y_test = data
        feature_names = X_train.columns.tolist()
        cohort_sample_size = len(X_train) + len(X_test)
        num_features_cnt = len(feature_names)

        fs_res = []
        for m_name, m_func in AVAILABLE_FS_METHODS.items():
            log_time_and_step(f"  Executing FS Method: {m_name}")
            try:
                r = m_func(X_train, y_train, X_test, y_test)
                if r:
                    r['method'] = m_name
                    fs_res.append(r)
            except Exception as e:
                log_time_and_step(f"    Error in FS method '{m_name}': {e}")

        subgroup_fs_rows = []
        for result in fs_res:
            method = result['method']
            indices = result['indices']
            scores = result['scores']
            score_type = result.get('score_type', 'Score')
            for rank_idx, (feat_idx, score) in enumerate(zip(indices, scores), start=1):
                if feat_idx < len(feature_names):
                    row = {
                        'Dataset': dataset_name,
                        'Subgroup': subgroup_name,
                        'Cohort Size': cohort_sample_size,
                        'Num Features': num_features_cnt,
                        'Method': method,
                        'Rank': rank_idx,
                        'Feature Name': feature_names[feat_idx],
                        'Score Type': score_type,
                        'Score': score
                    }
                    dataset_fs_rows.append(row)
                    subgroup_fs_rows.append(row)

        # Save incremental subgroup results to disk immediately
        if subgroup_fs_rows:
            sg_df = pd.DataFrame(subgroup_fs_rows)
            out_dirs = ["results", os.path.join(FS_DIR, "results"), "../../results"]
            for out_dir in out_dirs:
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    sg_file = os.path.join(out_dir, f"{dataset_name.lower()}_urban_rural_fs_results.csv")
                    if not os.path.exists(sg_file):
                        sg_df.to_csv(sg_file, index=False)
                    else:
                        sg_df.to_csv(sg_file, mode='a', header=False, index=False)
                except Exception as e:
                    pass
            log_time_and_step(f"   [Disk Saved] Incremental results for {dataset_name} ({subgroup_name}) saved to CSV.")

        del X_train, X_test, y_train, y_test
        gc.collect()

    del df
    gc.collect()

    # Save complete dataset results to dataset-specific CSV file
    if dataset_fs_rows:
        d_df = pd.DataFrame(dataset_fs_rows)
        out_dirs = ["results", os.path.join(FS_DIR, "results"), "../../results"]
        for out_dir in out_dirs:
            try:
                os.makedirs(out_dir, exist_ok=True)
                d_file = os.path.join(out_dir, f"{dataset_name.lower()}_urban_rural_fs_results.csv")
                d_df.to_csv(d_file, index=False)
                log_time_and_step(f"=========================================================")
                log_time_and_step(f"   SAVED {dataset_name.upper()} FS RESULTS TO: {d_file}")
                log_time_and_step(f"=========================================================")
            except Exception as e:
                log_time_and_step(f"Error saving dataset CSV: {e}")

    return dataset_fs_rows


def main():
    log_time_and_step("=================================================================")
    log_time_and_step("  TEXAS & NATIONWIDE URBAN VS. RURAL FEATURE SELECTION PIPELINE  ")
    log_time_and_step("=================================================================")
    log_time_and_step(f"Library Status -> LightGBM: {LGBM_AVAILABLE}, XGBoost: {XGB_AVAILABLE}, PyTorch: {TORCH_AVAILABLE}, Imblearn: {IMBLEARN_AVAILABLE}")

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(process_fs_for_dataset, d_name, paths): d_name
            for d_name, paths in DATASETS_TO_RUN.items()
        }
        for future in concurrent.futures.as_completed(futures):
            d_name = futures[future]
            try:
                rows = future.result()
                if rows:
                    all_results.extend(rows)
                    cum_df = pd.DataFrame(all_results)
                    out_dirs = ["results", os.path.join(FS_DIR, "results"), "../../results"]
                    for out_dir in out_dirs:
                        try:
                            os.makedirs(out_dir, exist_ok=True)
                            out_file = os.path.join(out_dir, "cohort_urban_rural_fs_results.csv")
                            cum_df.to_csv(out_file, index=False)
                            log_time_and_step(f"Saved cumulative results up to {d_name} to: {out_file}")
                        except Exception as e:
                            pass
            except Exception as e:
                log_time_and_step(f"Error in parallel FS for {d_name}: {e}")

    if all_results:
        fs_df = pd.DataFrame(all_results)

        # Summary of Top 5 Features per Dataset / Subgroup
        log_time_and_step("\n=================================================================")
        log_time_and_step("  FEATURE SELECTION SUMMARY (Top Features per Subgroup & Method)  ")
        log_time_and_step("=================================================================")
        top1 = fs_df[fs_df['Rank'] == 1]
        for (ds, sg, mthd), group in top1.groupby(['Dataset', 'Subgroup', 'Method']):
            top_feat = group['Feature Name'].values[0]
            top_score = group['Score'].values[0]
            print(f"  [{ds} | {sg} | {mthd}] Top Feature: {top_feat} (Score: {top_score:.4f})")

    log_time_and_step("\n=== Texas & Nationwide Urban/Rural Feature Selection Pipeline Completed ===")


if __name__ == "__main__":
    main()
