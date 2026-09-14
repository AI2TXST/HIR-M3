import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Add modeling directory to path
sys.path.append(os.path.dirname(__file__))

from models import (
    get_model_and_params, train_lgb, train_xgb, train_cb, train_rf, 
    train_hir, train_mlp, train_standard_transformer, train_saint, train_gbdt, train_logreg,
    split_features_by_level, LGBM_AVAILABLE, XGB_AVAILABLE, CATBOOST_AVAILABLE, 
    TORCH_AVAILABLE, IMBLEARN_AVAILABLE, LGBM_IMPORT_ERROR, XGB_IMPORT_ERROR, 
    CATBOOST_IMPORT_ERROR, TORCH_IMPORT_ERROR
)
from metrics import calculate_metrics, calculate_bootstrap_ci, find_optimal_threshold, plot_roc_pr_curves
from optimize_ensemble import optimize_ensemble_weights, evaluate_ensemble, run_ensemble_splits_experiment

import sys

for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

COHORTS_FILES = {
    "Nationwide Cohort": [
        "../data/processed_final_mergedDF_condensed.parquet",
        "data/processed_final_mergedDF_condensed.parquet",
        "../data/processed_final_mergedDF.parquet",
        "data/processed_final_mergedDF.parquet",
        "../data/processed_final_mergedDF_condensed.csv",
        "data/processed_final_mergedDF_condensed.csv"
    ],
    "Texas Cohort": [
        "../data/processed_final_mergedDF_TX.parquet",
        "data/processed_final_mergedDF_TX.parquet",
        "../data/processed_final_mergedDF_condensed_TX.parquet",
        "data/processed_final_mergedDF_condensed_TX.parquet",
        "../data/processed_final_mergedDF_TX.csv",
        "data/processed_final_mergedDF_TX.csv"
    ]
}

import gc

TARGET_COL = "ever_readmitted"

def load_and_prep_data(filepath, target_col=TARGET_COL):
    """
    Loads preprocessed dataset (preferring Parquet or CSV chunks to avoid OOM),
    applies downcasting and 20% sampling per chunk, and returns scaled train/test splits.
    """
    actual_path = None
    if isinstance(filepath, (list, tuple)):
        actual_path = next((p for p in filepath if os.path.exists(p)), None)
        filepath_str = actual_path if actual_path else str(filepath)
    else:
        filepath_str = str(filepath)
        if os.path.exists(filepath_str):
            actual_path = filepath_str

    if not actual_path:
        if "tx" in filepath_str.lower():
            candidates = [
                "../data/processed_final_mergedDF_condensed_TX.parquet",
                "data/processed_final_mergedDF_condensed_TX.parquet",
                "../data/processed_final_mergedDF_condensed_TX.csv",
                "data/processed_final_mergedDF_condensed_TX.csv"
            ]
        else:
            candidates = [
                "../data/processed_final_mergedDF_condensed.parquet",
                "data/processed_final_mergedDF_condensed.parquet",
                "../data/processed_final_mergedDF_condensed.csv",
                "data/processed_final_mergedDF_condensed.csv"
            ]
        actual_path = next((p for p in candidates if os.path.exists(p)), None)

    if not actual_path:
        logging.error(f"File not found for dataset path: {filepath}")
        return None

    filepath = actual_path
    is_texas = "tx" in filepath.lower() or "texas" in filepath.lower()

    if filepath.endswith('.parquet'):
        logging.info(f"Loading dataset from Parquet: {filepath}...")
        df = pd.read_parquet(filepath)
        initial_total_rows = len(df)
        if not is_texas:
            df = df.sample(frac=0.50, random_state=42).copy()
    else:
        logging.info(f"Loading dataset in memory-efficient chunks from CSV: {filepath}...")
        chunks = []
        initial_total_rows = 0

        for chunk in pd.read_csv(filepath, chunksize=100000, low_memory=False):
            initial_total_rows += len(chunk)

            if not is_texas:
                chunk = chunk.sample(frac=0.50, random_state=42).copy()

            int_cols = chunk.select_dtypes(include=['int64', 'int32']).columns
            for c in int_cols:
                c_min, c_max = chunk[c].min(), chunk[c].max()
                if c_min >= -128 and c_max <= 127:
                    chunk[c] = chunk[c].astype(np.int8)
                elif c_min >= -32768 and c_max <= 32767:
                    chunk[c] = chunk[c].astype(np.int16)
                else:
                    chunk[c] = chunk[c].astype(np.int32)
            
            float_cols = chunk.select_dtypes(include=['float64']).columns
            if len(float_cols) > 0:
                chunk[float_cols] = chunk[float_cols].astype(np.float32)

            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

    possible_targets = [target_col, 'ever_readmitted', 'READMISSION', 'Readmission']
    t_col = next((t for t in possible_targets if t in df.columns), None)
    
    if not t_col:
        logging.error(f"No target column found in {filepath}")
        return None

    total_sample_size = len(df)
    logging.info(f"Loaded modeling dataset shape: {df.shape} (from {initial_total_rows:,} raw rows)")

    drop_cols = [
        'BENE_ID', 'Beneficiary_ID', 'Assessment_Effective_Date', 'COUNTYFIPS',
        'Days_Cared_For', 'ever_deceased', 'NumVisits', 'DaysBetweenVisits',
        'PrevVisitDate', 'Last_Assessment_Date', t_col
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols and not c.lower().endswith('id')]

    # Downcast features to float32 immediately to reduce RAM
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

def tune_baseline_model(model_name, X_train, y_train):
    """
    Performs 3-fold Grid Search for a baseline ML model and returns best hyperparameters & best estimator.
    """
    logging.info(f"--- Hyperparameter Tuning for {model_name} ---")
    base_model, param_grid = get_model_and_params(model_name)
    if base_model is None or not param_grid:
        logging.warning(f"No parameter search grid for {model_name}. Using default parameters.")
        return {}, base_model

    try:
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='f1',
            cv=3,
            n_jobs=1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        logging.info(f"[{model_name}] Best Parameters: {grid_search.best_params_} (Best F1 CV: {grid_search.best_score_:.4f})")
        return grid_search.best_params_, grid_search.best_estimator_
    except Exception as e:
        logging.error(f"Hyperparameter tuning failed for {model_name} ({e}). Falling back to default.")
        return {}, base_model

def run_pipeline(selected_cohort="all"):
    logging.info("=== Starting Comprehensive Model Training Pipeline with Hyperparameter Tuning ===")
    logging.info(f"Detected Model Libraries -> LightGBM: {LGBM_AVAILABLE}, XGBoost: {XGB_AVAILABLE}, CatBoost: {CATBOOST_AVAILABLE}, PyTorch: {TORCH_AVAILABLE}")

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    all_model_results = []
    all_bootstrap_results = []

    if selected_cohort == "Texas":
        target_files = {k: v for k, v in COHORTS_FILES.items() if "Texas" in k}
    elif selected_cohort == "Nationwide":
        target_files = {k: v for k, v in COHORTS_FILES.items() if "Nationwide" in k}
    else:
        target_files = COHORTS_FILES

    for cohort_name, filepath in target_files.items():
        logging.info(f"\n--- Processing Cohort: {cohort_name} ---")
        data = load_and_prep_data(filepath)
        if data is None:
            continue

        X_train, X_test, y_train, y_test, feature_cols, cohort_sample_size = data
        w_train = np.ones_like(y_train, dtype=float)
        w_test = np.ones_like(y_test, dtype=float)

        best_hyperparams = {}
        y_prob_dict = {}

        # Baseline Models Tuning & Training (Parallel)
        from concurrent.futures import ThreadPoolExecutor

        def _task_lgb():
            if not LGBM_AVAILABLE: return "LightGBM", {}, None
            params, _ = tune_baseline_model("LightGBM", X_train, y_train)
            prob = train_lgb(X_train, y_train, w_train, X_test, y_test, params)
            return "LightGBM", params, prob

        def _task_xgb():
            if not XGB_AVAILABLE: return "XGBoost", {}, None
            params, _ = tune_baseline_model("XGBoost", X_train, y_train)
            prob = train_xgb(X_train, y_train, w_train, X_test, y_test, params)
            return "XGBoost", params, prob

        def _task_cb():
            if not CATBOOST_AVAILABLE: return "CatBoost", {}, None
            params, _ = tune_baseline_model("CatBoost", X_train, y_train)
            prob = train_cb(X_train, y_train, w_train, X_test, y_test, params)
            return "CatBoost", params, prob

        def _task_rf():
            params, _ = tune_baseline_model("Random Forest", X_train, y_train)
            prob = train_rf(X_train, y_train, w_train, X_test, y_test, params)
            return "Random Forest", params, prob

        def _task_gbdt():
            params, _ = tune_baseline_model("Gradient Boosting", X_train, y_train)
            prob = train_gbdt(X_train, y_train, w_train, X_test, y_test, params)
            return "Gradient Boosting", params, prob

        def _task_logreg():
            params, _ = tune_baseline_model("Logistic Regression", X_train, y_train)
            prob = train_logreg(X_train, y_train, w_train, X_test, y_test, params)
            return "Logistic Regression", params, prob

        baseline_tasks = [_task_lgb, _task_xgb, _task_cb, _task_rf, _task_gbdt, _task_logreg]
        for task in baseline_tasks:
            try:
                name, params, prob = task()
                if params:
                    best_hyperparams[name] = params
                if prob is not None:
                    y_prob_dict[name] = prob
            except Exception as e:
                logging.error(f"Baseline Error ({task}): {e}")

        # Save Tuned Hyperparameters JSON
        json_path = os.path.join(output_dir, f"{cohort_name.lower().replace(' ', '_')}_best_hyperparameters.json")
        with open(json_path, 'w') as f:
            json.dump(best_hyperparams, f, indent=4)
        logging.info(f"Saved tuned hyperparameters to: {json_path}")

        # 80/20 Train/Val split of X_train for validation threshold tuning (no test set leakage)
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
        w_tr = np.ones_like(y_tr, dtype=float)
        w_val = np.ones_like(y_val, dtype=float)

        # Predict validation probabilities to find threshold without test set leakage
        y_val_prob_dict = {}
        if LGBM_AVAILABLE and 'LightGBM' in y_prob_dict:
            y_val_prob_dict['LightGBM'] = train_lgb(X_tr, y_tr, w_tr, X_val, y_val, best_hyperparams.get('LightGBM', {}))
        if XGB_AVAILABLE and 'XGBoost' in y_prob_dict:
            y_val_prob_dict['XGBoost'] = train_xgb(X_tr, y_tr, w_tr, X_val, y_val, best_hyperparams.get('XGBoost', {}))
        if CATBOOST_AVAILABLE and 'CatBoost' in y_prob_dict:
            y_val_prob_dict['CatBoost'] = train_cb(X_tr, y_tr, w_tr, X_val, y_val, best_hyperparams.get('CatBoost', {}))
        if 'Random Forest' in y_prob_dict:
            y_val_prob_dict['Random Forest'] = train_rf(X_tr, y_tr, w_tr, X_val, y_val, best_hyperparams.get('Random Forest', {}))
        if 'Gradient Boosting' in y_prob_dict:
            y_val_prob_dict['Gradient Boosting'] = train_gbdt(X_tr, y_tr, w_tr, X_val, y_val, best_hyperparams.get('Gradient Boosting', {}))
        if 'Logistic Regression' in y_prob_dict:
            y_val_prob_dict['Logistic Regression'] = train_logreg(X_tr, y_tr, w_tr, X_val, y_val, best_hyperparams.get('Logistic Regression', {}))

        # Evaluate individual models on held-out test set
        model_results_cohort = []
        boot_results_cohort = []
        for model_name, y_prob in y_prob_dict.items():
            # Tune threshold on validation set (or fallback to test if val unavailable)
            if model_name in y_val_prob_dict and y_val_prob_dict[model_name] is not None:
                thresh, _ = find_optimal_threshold(y_val, y_val_prob_dict[model_name], metric='f1')
            else:
                thresh = 0.5

            m = calculate_metrics(y_test, y_prob, threshold=thresh)
            m['Cohort'] = cohort_name
            m['Cohort Size'] = cohort_sample_size
            m['Num Features'] = len(feature_cols)
            m['Model'] = model_name
            all_model_results.append(m)
            model_results_cohort.append(m)

            # Compute Bootstrap CIs
            boot_summary = calculate_bootstrap_ci(y_test, y_prob, threshold=thresh, n_bootstrap=200)
            for m_name, b_data in boot_summary.items():
                b_item = {
                    'Cohort': cohort_name,
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



        # Save cohort-specific output files
        prefix = cohort_name.replace(" ", "_")
        if model_results_cohort:
            c_df = pd.DataFrame(model_results_cohort)
            c_path = os.path.join(output_dir, f"{prefix}_modeling_results.csv")
            c_df.to_csv(c_path, index=False)
            logging.info(f"Saved {cohort_name} modeling results to {c_path}")

        if boot_results_cohort:
            b_df = pd.DataFrame(boot_results_cohort)
            b_path = os.path.join(output_dir, f"{prefix}_bootstrap_results.csv")
            b_df.to_csv(b_path, index=False)
            logging.info(f"Saved {cohort_name} bootstrap results to {b_path}")

        del X_train, X_test, y_train, y_test
        gc.collect()

    # Export consolidated summary across all cohorts
    if all_model_results:
        results_df = pd.DataFrame(all_model_results)
        results_path = os.path.join(output_dir, "modeling_results_all.csv")
        results_df.to_csv(results_path, index=False)
        logging.info(f"\nSaved consolidated modeling results to {results_path}")

    if all_bootstrap_results:
        boot_df = pd.DataFrame(all_bootstrap_results)
        boot_path = os.path.join(output_dir, "bootstrap_results_all.csv")
        boot_df.to_csv(boot_path, index=False)
        logging.info(f"Saved consolidated bootstrap results to {boot_path}")

    logging.info("=== Modeling Pipeline Finished Successfully ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Cohort Modeling Pipeline")
    parser.add_argument("--cohort", choices=["Texas", "Nationwide", "all"], default="all",
                        help="Specify cohort to train (Texas, Nationwide, or all)")
    args = parser.parse_args()
    run_pipeline(args.cohort)
