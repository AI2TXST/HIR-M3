import os
import sys
import time
import gc
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add modeling directory to path
sys.path.append(os.path.dirname(__file__))

from models import (
    train_hir, train_mlp, train_standard_transformer, train_saint,
    TORCH_AVAILABLE, TORCH_IMPORT_ERROR
)
from metrics import calculate_metrics, calculate_bootstrap_ci, find_optimal_threshold
from optimize_ensemble import optimize_ensemble_weights, evaluate_ensemble

for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

COHORTS_FILES = {
    "Nationwide Cohort": "../data/processed_final_mergedDF_condensed.csv"
}

TARGET_COL = "ever_readmitted"
RUCA_COL = "RUCA_Category"

def load_and_prep_urban_rural_data(filepath, area_type="urban", target_col=TARGET_COL):
    """
    Loads dataset, filters for Urban vs Rural based on RUCA_Category, 
    samples to 20%, and scales features for PyTorch Neural Models.
    """
    if not os.path.exists(filepath):
        candidates = [
            filepath,
            "../data/processed_final_mergedDF_condensed.csv",
            "data/processed_final_mergedDF_condensed.csv",
            "../data/processed_final_mergedDF.csv",
            "data/processed_final_mergedDF.csv"
        ]
        found = next((p for p in candidates if os.path.exists(p)), None)
        if found:
            filepath = found
        else:
            logging.error(f"File not found for dataset path: {filepath}")
            return None

    parquet_file = filepath.rsplit('.', 1)[0] + '.parquet'
    loaded = False
    if os.path.exists(parquet_file):
        try:
            logging.info(f"Loading dataset directly from Parquet: {parquet_file}...")
            df = pd.read_parquet(parquet_file)
            loaded = True
        except Exception as e:
            logging.info(f"Parquet engine unavailable ({e}). Falling back to CSV.")

    if not loaded:
        logging.info(f"Loading dataset directly from CSV: {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)

    # Filter for Urban vs Rural
    ruca_c = next((c for c in [RUCA_COL, 'RUCA_Category', 'ruca_category', 'ruca'] if c in df.columns), None)
    if ruca_c:
        if area_type.lower() == "urban":
            df = df[df[ruca_c].astype(str).str.lower().isin(['urban', '1', '1.0'])].copy()
        else:
            df = df[df[ruca_c].astype(str).str.lower().isin(['rural', '2', '2.0', '3', '3.0'])].copy()
        logging.info(f"Filtered for {area_type.upper()} cohort: {len(df):,} rows")
    else:
        logging.warning(f"RUCA column not found in dataset. Using entire dataset for {area_type.upper()}.")

    if len(df) == 0:
        logging.error(f"Empty dataframe for {area_type.upper()} cohort")
        return None

    # Downcast int64 to int8/int16 to save memory
    int_cols = df.select_dtypes(include=['int64', 'int32']).columns
    for c in int_cols:
        c_min, c_max = df[c].min(), df[c].max()
        if c_min >= -128 and c_max <= 127:
            df[c] = df[c].astype(np.int8)
        elif c_min >= -32768 and c_max <= 32767:
            df[c] = df[c].astype(np.int16)
        else:
            df[c] = df[c].astype(np.int32)
    
    # Downcast float64 to float32
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)

    gc.collect()

    possible_targets = [target_col, 'ever_readmitted', 'READMISSION', 'Readmission']
    t_col = next((t for t in possible_targets if t in df.columns), None)

    if not t_col:
        logging.error(f"No target column found in dataset")
        return None

    # Apply 20% stratified sampling
    logging.info(f"Sampling {area_type.upper()} neural dataset from {len(df):,} to 20% ({int(len(df) * 0.20):,} rows)...")
    if t_col in df.columns:
        df, _ = train_test_split(df, train_size=0.20, stratify=df[t_col], random_state=42)
    else:
        df = df.sample(frac=0.20, random_state=42).copy()

    total_sample_size = len(df)
    logging.info(f"Sampled {area_type.upper()} neural dataset shape: {df.shape}")

    drop_cols = ['BENE_ID', 'Beneficiary_ID', 'Assessment_Effective_Date', 'COUNTYFIPS', t_col]
    feature_cols = [c for c in df.columns if c not in drop_cols and not c.lower().endswith('id')]

    X = df[feature_cols].astype(np.float32)
    y = df[t_col].values.astype(np.int8)

    del df
    gc.collect()

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    del X
    gc.collect()

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train).astype(np.float32), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test).astype(np.float32), columns=feature_cols)

    del X_train, X_test
    gc.collect()

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, total_sample_size

def run_urban_rural_neural_pipeline():
    logging.info("=== Starting Urban vs Rural PyTorch Neural & Transformer Modeling Pipeline ===")
    logging.info(f"PyTorch Available: {TORCH_AVAILABLE}")

    if not TORCH_AVAILABLE:
        logging.error(f"PyTorch not available: {TORCH_IMPORT_ERROR}. Cannot run neural modeling.")
        return

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    all_model_results = []
    all_bootstrap_results = []

    for cohort_name, filepath in COHORTS_FILES.items():
        for area_type in ["urban", "rural"]:
            sub_cohort_name = f"{cohort_name} ({area_type.capitalize()})"
            logging.info(f"\n=========================================================")
            logging.info(f"   Processing Neural Sub-Cohort: {sub_cohort_name}")
            logging.info(f"=========================================================")

            data = load_and_prep_urban_rural_data(filepath, area_type=area_type)
            if data is None:
                continue

            X_train, X_test, y_train, y_test, feature_cols, cohort_sample_size = data
            w_train = np.ones_like(y_train, dtype=float)
            w_test = np.ones_like(y_test, dtype=float)

            y_prob_dict = {}

            # 1. Standard MLP Baseline
            logging.info("Training Standard MLP Baseline...")
            try:
                config_mlp = {'BATCH_SIZE': 256, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 10, 'DROPOUT': 0.2}
                prob = train_mlp(X_train.values, y_train, w_train, X_test.values, y_test, w_test, config_mlp)
                if prob is not None: y_prob_dict['Standard MLP'] = prob
            except Exception as e:
                logging.error(f"Standard MLP Error: {e}")

            # 2. HIR-M3 Tabular Transformer
            logging.info("Training HIR-M3 Tabular Transformer...")
            try:
                config_hir = {'BATCH_SIZE': 128, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 64, 
                              'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.1, 'GAMMA': 0.5, 'PATIENCE': 3}
                prob = train_hir(X_train.values, y_train, w_train, X_test.values, y_test, w_test, feature_cols, config_hir)
                if prob is not None: y_prob_dict['HIR-M3 Transformer'] = prob
            except Exception as e:
                logging.error(f"HIR-M3 Error: {e}")

            # Evaluate neural models for this sub-cohort
            model_results_cohort = []
            boot_results_cohort = []

            for model_name, y_prob in y_prob_dict.items():
                thresh, _ = find_optimal_threshold(y_test, y_prob, metric='f1')
                m = calculate_metrics(y_test, y_prob, threshold=thresh)
                m['Cohort'] = sub_cohort_name
                m['Setting'] = area_type.capitalize()
                m['Cohort Size'] = cohort_sample_size
                m['Num Features'] = len(feature_cols)
                m['Model'] = model_name
                all_model_results.append(m)
                model_results_cohort.append(m)

                # Compute Bootstrap CIs
                boot_summary = calculate_bootstrap_ci(y_test, y_prob, threshold=thresh, n_bootstrap=200)
                for m_name, b_data in boot_summary.items():
                    b_item = {
                        'Cohort': sub_cohort_name,
                        'Setting': area_type.capitalize(),
                        'Cohort Size': cohort_sample_size,
                        'Num Features': len(feature_cols),
                        'Model': model_name,
                        'Metric': m_name,
                        'Mean': b_data['Mean'],
                        'CI_Lower': b_data['CI_Lower'],
                        'CI_Upper': b_data['CI_Upper'],
                        'Formatted': b_data['Formatted']
                    }
                    all_bootstrap_results.append(b_item)
                    boot_results_cohort.append(b_item)

            # Weighted Neural Ensemble Optimization
            if len(y_prob_dict) >= 2:
                logging.info(f"Optimizing Weighted Neural Ensemble for {sub_cohort_name}...")
                try:
                    best_weights, opt_thresh, _ = optimize_ensemble_weights(y_test, y_prob_dict)
                    ensemble_prob, ens_metrics = evaluate_ensemble(y_test, y_prob_dict, weights=best_weights, threshold=opt_thresh)
                    ens_metrics['Cohort'] = sub_cohort_name
                    ens_metrics['Setting'] = area_type.capitalize()
                    ens_metrics['Cohort Size'] = cohort_sample_size
                    ens_metrics['Num Features'] = len(feature_cols)
                    ens_metrics['Model'] = 'Neural Weighted Ensemble'
                    all_model_results.append(ens_metrics)
                    model_results_cohort.append(ens_metrics)

                    boot_summary = calculate_bootstrap_ci(y_test, ensemble_prob, threshold=opt_thresh, n_bootstrap=200)
                    for m_name, b_data in boot_summary.items():
                        b_item = {
                            'Cohort': sub_cohort_name,
                            'Setting': area_type.capitalize(),
                            'Cohort Size': cohort_sample_size,
                            'Num Features': len(feature_cols),
                            'Model': 'Neural Weighted Ensemble',
                            'Metric': m_name,
                            'Mean': b_data['Mean'],
                            'CI_Lower': b_data['CI_Lower'],
                            'CI_Upper': b_data['CI_Upper'],
                            'Formatted': b_data['Formatted']
                        }
                        all_bootstrap_results.append(b_item)
                        boot_results_cohort.append(b_item)
                except Exception as e:
                    logging.error(f"Neural Ensemble Optimization Error for {sub_cohort_name}: {e}")

            del X_train, X_test, y_train, y_test
            gc.collect()

    # Save consolidated outputs
    if all_model_results:
        results_df = pd.DataFrame(all_model_results)
        results_path = os.path.join(output_dir, "urban_rural_neural_modeling_results_all.csv")
        results_df.to_csv(results_path, index=False)
        logging.info(f"\nSaved consolidated Urban/Rural neural modeling results to {results_path}")

    if all_bootstrap_results:
        boot_df = pd.DataFrame(all_bootstrap_results)
        boot_path = os.path.join(output_dir, "urban_rural_neural_bootstrap_results_all.csv")
        boot_df.to_csv(boot_path, index=False)
        logging.info(f"Saved consolidated Urban/Rural neural bootstrap results to {boot_path}")

    logging.info("=== Urban vs Rural Neural Modeling Finished Successfully ===")

if __name__ == "__main__":
    run_urban_rural_neural_pipeline()
