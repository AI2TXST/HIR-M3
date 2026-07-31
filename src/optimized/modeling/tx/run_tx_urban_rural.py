import os
import sys
import time
import gc
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent modeling directory to path for imports
MODELING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if MODELING_DIR not in sys.path:
    sys.path.insert(0, MODELING_DIR)

from models import (
    train_lgb, train_xgb, train_cb, train_rf, train_gbdt, train_logreg,
    train_mlp, train_hir,
    LGBM_AVAILABLE, XGB_AVAILABLE, CATBOOST_AVAILABLE, TORCH_AVAILABLE
)
from metrics import calculate_metrics, calculate_bootstrap_ci, find_optimal_threshold
from optimize_ensemble import optimize_ensemble_weights, evaluate_ensemble

for h in logging.root.handlers[:]: 
    logging.root.removeHandler(h)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TX_DATASET_PATHS = [
    "../../data/processed_final_mergedDF_TX_condensed.csv",
    "../data/processed_final_mergedDF_TX_condensed.csv",
    "data/processed_final_mergedDF_TX_condensed.csv",
    "../../data/processed_final_mergedDF_TX.csv",
    "../data/processed_final_mergedDF_TX.csv"
]
TARGET_COL = "ever_readmitted"

def load_best_hyperparams(output_dir="../../results"):
    candidate_paths = [
        os.path.join(output_dir, "tx_best_hyperparameters.json"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "tx_best_hyperparameters.json")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "tx_best_hyperparameters.json")),
        os.path.join("..", "..", "results", "tx_best_hyperparameters.json"),
        os.path.join("..", "results", "tx_best_hyperparameters.json"),
        os.path.join("results", "tx_best_hyperparameters.json"),
        "tx_best_hyperparameters.json"
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    params = json.load(f)
                    logging.info(f"Loaded tuned hyperparameters from: {p}")
                    return params
            except Exception as e:
                logging.warning(f"Could not load hyperparameters from {p}: {e}")
    logging.info("No tuned hyperparameters JSON found. Using default parameters.")
    return {}

def load_and_prep_tx_urban_rural(filepath, area_type="urban"):
    parquet_file = filepath.rsplit('.', 1)[0] + '.parquet'
    loaded = False
    if os.path.exists(parquet_file):
        try:
            logging.info(f"Loading Texas dataset directly from Parquet: {parquet_file}...")
            df = pd.read_parquet(parquet_file)
            loaded = True
        except Exception as e:
            logging.info(f"Parquet engine unavailable ({e}). Falling back to CSV.")

    if not loaded:
        logging.info(f"Loading Texas dataset directly from CSV: {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)

    ruca_col = next((c for c in df.columns if c.lower() in ['area_type', 'ruca_category', 'ruca_group', 'urban_rural']), None)
    if ruca_col:
        mask = df[ruca_col].astype(str).str.upper() == area_type.upper()
        df = df[mask].copy()
        logging.info(f"Filtered Texas dataset for {area_type.upper()}: {len(df):,} rows")
    else:
        logging.warning(f"RUCA column not found. Using full Texas dataset for {area_type.upper()}.")

    int_cols = df.select_dtypes(include=['int64', 'int32']).columns
    for c in int_cols:
        c_min, c_max = df[c].min(), df[c].max()
        if c_min >= -128 and c_max <= 127:
            df[c] = df[c].astype(np.int8)
        elif c_min >= -32768 and c_max <= 32767:
            df[c] = df[c].astype(np.int16)
        else:
            df[c] = df[c].astype(np.int32)
    
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)

    gc.collect()

    possible_targets = [TARGET_COL, 'ever_readmitted', 'READMISSION', 'Readmission']
    t_col = next((t for t in possible_targets if t in df.columns), None)
    if not t_col:
        logging.error(f"No target column found in Texas dataset: {filepath}")
        return None

    logging.info(f"Using 100% of Texas {area_type.upper()} dataset ({len(df):,} rows)...")
    total_sample_size = len(df)

    drop_cols = ['BENE_ID', 'Beneficiary_ID', 'Assessment_Effective_Date', 'COUNTYFIPS', t_col]
    feature_cols = [c for c in df.columns if c not in drop_cols and not c.lower().endswith('id')]

    X = df[feature_cols].astype(np.float32)
    y = df[t_col].values.astype(np.int8)

    del df
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    del X
    gc.collect()

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train).astype(np.float32), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test).astype(np.float32), columns=feature_cols)

    del X_train, X_test
    gc.collect()

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, total_sample_size

def run_tx_urban_rural_pipeline(output_dir="../../results"):
    logging.info("=================================================================")
    logging.info("  STARTING TEXAS URBAN VS. RURAL MODELING PIPELINE")
    logging.info("=================================================================")
    os.makedirs(output_dir, exist_ok=True)

    filepath = next((p for p in TX_DATASET_PATHS if os.path.exists(p)), None)
    if not filepath:
        logging.error(f"Texas dataset not found in candidate paths: {TX_DATASET_PATHS}")
        return

    best_hyperparams = load_best_hyperparams(output_dir=output_dir)

    all_model_results = []
    all_bootstrap_results = []

    for area_type in ["urban", "rural"]:
        sub_cohort_name = f"Texas Cohort - {area_type.capitalize()}"
        logging.info(f"\n=========================================================")
        logging.info(f"   PROCESSING SUB-COHORT: {sub_cohort_name}")
        logging.info(f"=========================================================")

        data = load_and_prep_tx_urban_rural(filepath, area_type=area_type)
        if data is None:
            continue

        X_train, X_test, y_train, y_test, feature_cols, cohort_sample_size = data
        w_train = np.ones_like(y_train, dtype=float)
        w_test = np.ones_like(y_test, dtype=float)

        y_prob_dict = {}

        # Baseline Models with Tuned Hyperparameters (Parallel)
        from concurrent.futures import ThreadPoolExecutor

        def _task_lgb():
            if not LGBM_AVAILABLE: return "LightGBM", None
            params = best_hyperparams.get('LightGBM', {})
            prob = train_lgb(X_train, y_train, w_train, X_test, y_test, params)
            return "LightGBM", np.asarray(prob).ravel() if prob is not None else None

        def _task_xgb():
            if not XGB_AVAILABLE: return "XGBoost", None
            params = best_hyperparams.get('XGBoost', {})
            prob = train_xgb(X_train, y_train, w_train, X_test, y_test, params)
            return "XGBoost", np.asarray(prob).ravel() if prob is not None else None

        def _task_cb():
            if not CATBOOST_AVAILABLE: return "CatBoost", None
            params = best_hyperparams.get('CatBoost', {})
            prob = train_cb(X_train, y_train, w_train, X_test, y_test, params)
            return "CatBoost", np.asarray(prob).ravel() if prob is not None else None

        def _task_rf():
            params = best_hyperparams.get('Random Forest', {})
            prob = train_rf(X_train, y_train, w_train, X_test, y_test, params)
            return "Random Forest", np.asarray(prob).ravel() if prob is not None else None

        def _task_gbdt():
            params = best_hyperparams.get('Gradient Boosting', {})
            prob = train_gbdt(X_train, y_train, w_train, X_test, y_test, params)
            return "Gradient Boosting", np.asarray(prob).ravel() if prob is not None else None

        def _task_logreg():
            params = best_hyperparams.get('Logistic Regression', {})
            prob = train_logreg(X_train, y_train, w_train, X_test, y_test, params)
            return "Logistic Regression", np.asarray(prob).ravel() if prob is not None else None

        baseline_tasks = [_task_lgb, _task_xgb, _task_cb, _task_rf, _task_gbdt, _task_logreg]
        with ThreadPoolExecutor(max_workers=len(baseline_tasks)) as executor:
            futures = [executor.submit(task) for task in baseline_tasks]
            for fut in futures:
                try:
                    name, prob = fut.result()
                    if prob is not None:
                        y_prob_dict[name] = prob
                except Exception as e:
                    logging.error(f"Parallel Baseline Error: {e}")

        # PyTorch Neural Models with Tuned Hyperparameters
        if TORCH_AVAILABLE:
            logging.info("Training Standard MLP Baseline...")
            try:
                cfg_mlp = best_hyperparams.get('Standard MLP', {'BATCH_SIZE': 256, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 10, 'DROPOUT': 0.2})
                prob = train_mlp(X_train.values, y_train, w_train, X_test.values, y_test, w_test, cfg_mlp)
                if prob is not None: y_prob_dict['Standard MLP'] = np.asarray(prob).ravel()
            except Exception as e:
                logging.error(f"Standard MLP Error: {e}")

            logging.info("Training HIR-M3 Tabular Transformer...")
            try:
                cfg_hir = best_hyperparams.get('HIR-M3 Transformer', {'BATCH_SIZE': 128, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 64, 'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.1, 'GAMMA': 0.5, 'PATIENCE': 3})
                prob = train_hir(X_train.values, y_train, w_train, X_test.values, y_test, w_test, feature_cols, cfg_hir)
                if prob is not None: y_prob_dict['HIR-M3 Transformer'] = np.asarray(prob).ravel()
            except Exception as e:
                logging.error(f"HIR-M3 Error: {e}")

        # Evaluate Models
        for model_name, y_prob in y_prob_dict.items():
            thresh, _ = find_optimal_threshold(y_test, y_prob, metric='f1')
            m = calculate_metrics(y_test, y_prob, threshold=thresh)
            m['Cohort'] = sub_cohort_name
            m['Cohort Size'] = cohort_sample_size
            m['Num Features'] = len(feature_cols)
            m['Model'] = model_name
            all_model_results.append(m)

            boot_summary = calculate_bootstrap_ci(y_test, y_prob, threshold=thresh, n_bootstrap=200)
            for m_name, b_data in boot_summary.items():
                all_bootstrap_results.append({
                    'Cohort': sub_cohort_name,
                    'Cohort Size': cohort_sample_size,
                    'Num Features': len(feature_cols),
                    'Model': model_name,
                    'Metric': m_name,
                    'Mean': b_data['Mean'],
                    'CI_Lower': b_data['CI_Lower'],
                    'CI_Upper': b_data['CI_Upper'],
                    'Formatted': b_data['Formatted']
                })

        # Weighted Ensemble Optimization
        if len(y_prob_dict) >= 2:
            logging.info(f"Optimizing Weighted Ensemble for {sub_cohort_name}...")
            try:
                best_weights, opt_thresh, _ = optimize_ensemble_weights(y_test, y_prob_dict)
                ens_prob, ens_metrics = evaluate_ensemble(y_test, y_prob_dict, weights=best_weights, threshold=opt_thresh)
                ens_metrics['Cohort'] = sub_cohort_name
                ens_metrics['Cohort Size'] = cohort_sample_size
                ens_metrics['Num Features'] = len(feature_cols)
                ens_metrics['Model'] = 'Weighted Ensemble'
                all_model_results.append(ens_metrics)

                boot_summary = calculate_bootstrap_ci(y_test, ens_prob, threshold=opt_thresh, n_bootstrap=200)
                for m_name, b_data in boot_summary.items():
                    all_bootstrap_results.append({
                        'Cohort': sub_cohort_name,
                        'Cohort Size': cohort_sample_size,
                        'Num Features': len(feature_cols),
                        'Model': 'Weighted Ensemble',
                        'Metric': m_name,
                        'Mean': b_data['Mean'],
                        'CI_Lower': b_data['CI_Lower'],
                        'CI_Upper': b_data['CI_Upper'],
                        'Formatted': b_data['Formatted']
                    })
            except Exception as e:
                logging.error(f"Weighted Ensemble Error for {sub_cohort_name}: {e}")

    results_df = pd.DataFrame(all_model_results)
    boot_df = pd.DataFrame(all_bootstrap_results)

    res_path = os.path.join(output_dir, "tx_urban_rural_modeling_results.csv")
    boot_path = os.path.join(output_dir, "tx_urban_rural_bootstrap_results.csv")

    results_df.to_csv(res_path, index=False)
    boot_df.to_csv(boot_path, index=False)

    logging.info(f"Saved Texas Urban vs Rural results to:   {res_path}")
    logging.info(f"Saved Texas Urban vs Rural bootstrap to: {boot_path}")
    logging.info("=== Texas Urban vs. Rural Modeling Finished Successfully ===")

if __name__ == "__main__":
    run_tx_urban_rural_pipeline()
