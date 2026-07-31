import os
import sys
import time
import gc
import copy
import json
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Add parent modeling directory to path for imports
MODELING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if MODELING_DIR not in sys.path:
    sys.path.insert(0, MODELING_DIR)

from models import (
    train_lgb, train_xgb, train_cb, train_rf, train_gbdt, train_logreg,
    train_mlp, train_hir, get_model_and_params,
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

def load_and_prep_tx_data():
    filepath = next((p for p in TX_DATASET_PATHS if os.path.exists(p)), None)
    if not filepath:
        logging.error(f"Texas dataset not found in candidate paths: {TX_DATASET_PATHS}")
        return None

    parquet_file = filepath.rsplit('.', 1)[0] + '.parquet'
    loaded = False
    if os.path.exists(parquet_file):
        try:
            logging.info(f"Loading Texas dataset directly from Parquet: {parquet_file}...")
            df = pd.read_parquet(parquet_file)
            loaded = True
        except Exception as e:
            logging.info(f"Parquet load failed ({e}). Falling back to CSV.")

    if not loaded:
        logging.info(f"Loading Texas dataset directly from CSV: {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)

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

    logging.info(f"Using 100% of Texas dataset ({len(df):,} rows)...")
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

def tune_mlp(X_train, y_train, w_train, X_val, y_val, w_val):
    """
    Grid search for Standard MLP hyperparameters.
    """
    logging.info("--- Hyperparameter Tuning for Standard MLP ---")
    param_grid = [
        {'BATCH_SIZE': 256, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 10, 'DROPOUT': 0.1},
        {'BATCH_SIZE': 256, 'HIDDEN_DIM': 256, 'LR': 1e-3, 'EPOCHS': 10, 'DROPOUT': 0.2},
        {'BATCH_SIZE': 128, 'HIDDEN_DIM': 128, 'LR': 5e-4, 'EPOCHS': 10, 'DROPOUT': 0.2},
    ]
    best_f1 = -1.0
    best_config = param_grid[0]

    for cfg in param_grid:
        try:
            prob = train_mlp(X_train, y_train, w_train, X_val, y_val, w_val, cfg)
            if prob is not None:
                opt_th, _ = find_optimal_threshold(y_val, prob, metric='f1')
                m = calculate_metrics(y_val, prob, threshold=opt_th)
                if m['F1_Score'] > best_f1:
                    best_f1 = m['F1_Score']
                    best_config = cfg
        except Exception as e:
            logging.error(f"MLP Config Search Error ({cfg}): {e}")

    logging.info(f"[Standard MLP] Best Parameters: {best_config} (Validation F1: {best_f1:.4f})")
    return best_config

def tune_hir(X_train, y_train, w_train, X_val, y_val, w_val, feature_cols):
    """
    Grid search for HIR-M3 Tabular Transformer hyperparameters.
    """
    logging.info("--- Hyperparameter Tuning for HIR-M3 Tabular Transformer ---")
    param_grid = [
        {'BATCH_SIZE': 128, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 64, 'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.1, 'GAMMA': 0.5, 'PATIENCE': 3},
        {'BATCH_SIZE': 128, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.05, 'GAMMA': 0.5, 'PATIENCE': 3},
        {'BATCH_SIZE': 256, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 64, 'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.1, 'GAMMA': 0.3, 'PATIENCE': 3},
    ]
    best_f1 = -1.0
    best_config = param_grid[0]

    for cfg in param_grid:
        try:
            prob = train_hir(X_train, y_train, w_train, X_val, y_val, w_val, feature_cols, cfg)
            if prob is not None:
                opt_th, _ = find_optimal_threshold(y_val, prob, metric='f1')
                m = calculate_metrics(y_val, prob, threshold=opt_th)
                if m['F1_Score'] > best_f1:
                    best_f1 = m['F1_Score']
                    best_config = cfg
        except Exception as e:
            logging.error(f"HIR Config Search Error ({cfg}): {e}")

    logging.info(f"[HIR-M3 Transformer] Best Parameters: {best_config} (Validation F1: {best_f1:.4f})")
    return best_config

def run_tx_modeling_pipeline(output_dir="../../results"):
    logging.info("=================================================================")
    logging.info("  STARTING TEXAS COHORT HYPERPARAMETER TUNING & MODELING PIPELINE")
    logging.info("=================================================================")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    data = load_and_prep_tx_data()
    if data is None:
        logging.error("Failed to prepare Texas dataset. Aborting pipeline.")
        return

    X_train, X_test, y_train, y_test, feature_cols, cohort_sample_size = data
    w_train = np.ones_like(y_train, dtype=float)
    w_test = np.ones_like(y_test, dtype=float)

    # Validation split for neural grid tuning
    X_tr_val, X_val, y_tr_val, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    w_tr_val = np.ones_like(y_tr_val, dtype=float)
    w_val = np.ones_like(y_val, dtype=float)

    best_hyperparams = {}
    y_prob_dict = {}

    # Load pre-tuned parameters if available
    pre_loaded_params = {}
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
                    pre_loaded_params = json.load(f)
                    logging.info(f"Successfully loaded pre-tuned hyperparameters from: {p}")
                    break
            except Exception as e:
                logging.warning(f"Could not load pre-tuned hyperparameters from {p}: {e}")

    best_hyperparams = copy.deepcopy(pre_loaded_params)

    # -------------------------------------------------------------
    # 1. Hyperparameter Tuning & Training Baseline ML Models (Parallel)
    # -------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor

    def _task_lgb():
        if not LGBM_AVAILABLE: return "LightGBM", {}, None
        if "LightGBM" in pre_loaded_params and pre_loaded_params["LightGBM"]:
            params = pre_loaded_params["LightGBM"]
            logging.info(f"[LightGBM] Using pre-tuned hyperparameters: {params}")
        else:
            params, _ = tune_baseline_model("LightGBM", X_train, y_train)
        prob = train_lgb(X_train, y_train, w_train, X_test, y_test, params)
        return "LightGBM", params, np.asarray(prob).ravel() if prob is not None else None

    def _task_xgb():
        if not XGB_AVAILABLE: return "XGBoost", {}, None
        if "XGBoost" in pre_loaded_params and pre_loaded_params["XGBoost"]:
            params = pre_loaded_params["XGBoost"]
            logging.info(f"[XGBoost] Using pre-tuned hyperparameters: {params}")
        else:
            params, _ = tune_baseline_model("XGBoost", X_train, y_train)
        prob = train_xgb(X_train, y_train, w_train, X_test, y_test, params)
        return "XGBoost", params, np.asarray(prob).ravel() if prob is not None else None

    def _task_cb():
        if not CATBOOST_AVAILABLE: return "CatBoost", {}, None
        if "CatBoost" in pre_loaded_params and pre_loaded_params["CatBoost"]:
            params = pre_loaded_params["CatBoost"]
            logging.info(f"[CatBoost] Using pre-tuned hyperparameters: {params}")
        else:
            params, _ = tune_baseline_model("CatBoost", X_train, y_train)
        prob = train_cb(X_train, y_train, w_train, X_test, y_test, params)
        return "CatBoost", params, np.asarray(prob).ravel() if prob is not None else None

    def _task_rf():
        if "Random Forest" in pre_loaded_params and pre_loaded_params["Random Forest"]:
            params = pre_loaded_params["Random Forest"]
            logging.info(f"[Random Forest] Using pre-tuned hyperparameters: {params}")
        else:
            params, _ = tune_baseline_model("Random Forest", X_train, y_train)
        prob = train_rf(X_train, y_train, w_train, X_test, y_test, params)
        return "Random Forest", params, np.asarray(prob).ravel() if prob is not None else None

    def _task_gbdt():
        if "Gradient Boosting" in pre_loaded_params and pre_loaded_params["Gradient Boosting"]:
            params = pre_loaded_params["Gradient Boosting"]
            logging.info(f"[Gradient Boosting] Using pre-tuned hyperparameters: {params}")
        else:
            params, _ = tune_baseline_model("Gradient Boosting", X_train, y_train)
        prob = train_gbdt(X_train, y_train, w_train, X_test, y_test, params)
        return "Gradient Boosting", params, np.asarray(prob).ravel() if prob is not None else None

    def _task_logreg():
        if "Logistic Regression" in pre_loaded_params and pre_loaded_params["Logistic Regression"]:
            params = pre_loaded_params["Logistic Regression"]
            logging.info(f"[Logistic Regression] Using pre-tuned hyperparameters: {params}")
        else:
            params, _ = tune_baseline_model("Logistic Regression", X_train, y_train)
        prob = train_logreg(X_train, y_train, w_train, X_test, y_test, params)
        return "Logistic Regression", params, np.asarray(prob).ravel() if prob is not None else None

    baseline_tasks = [_task_lgb, _task_xgb, _task_cb, _task_rf, _task_gbdt, _task_logreg]
    with ThreadPoolExecutor(max_workers=len(baseline_tasks)) as executor:
        futures = [executor.submit(task) for task in baseline_tasks]
        for fut in futures:
            try:
                name, params, prob = fut.result()
                if params:
                    best_hyperparams[name] = params
                if prob is not None:
                    y_prob_dict[name] = prob
            except Exception as e:
                logging.error(f"Parallel Baseline Error: {e}")

    # -------------------------------------------------------------
    # 2. Hyperparameter Tuning & Training PyTorch Neural Models
    # -------------------------------------------------------------
    if TORCH_AVAILABLE:
        if "Standard MLP" in pre_loaded_params and pre_loaded_params["Standard MLP"]:
            mlp_config = pre_loaded_params["Standard MLP"]
            logging.info(f"[Standard MLP] Using pre-tuned hyperparameters: {mlp_config}")
        else:
            mlp_config = tune_mlp(X_tr_val.values, y_tr_val, w_tr_val, X_val.values, y_val, w_val)
        best_hyperparams['Standard MLP'] = mlp_config
        try:
            prob = train_mlp(X_train.values, y_train, w_train, X_test.values, y_test, w_test, mlp_config)
            if prob is not None: y_prob_dict['Standard MLP'] = np.asarray(prob).ravel()
        except Exception as e:
            logging.error(f"Standard MLP Training Error: {e}")

        if "HIR-M3 Transformer" in pre_loaded_params and pre_loaded_params["HIR-M3 Transformer"]:
            hir_config = pre_loaded_params["HIR-M3 Transformer"]
            logging.info(f"[HIR-M3 Transformer] Using pre-tuned hyperparameters: {hir_config}")
        else:
            hir_config = tune_hir(X_tr_val.values, y_tr_val, w_tr_val, X_val.values, y_val, w_val, feature_cols)
        best_hyperparams['HIR-M3 Transformer'] = hir_config
        try:
            prob = train_hir(X_train.values, y_train, w_train, X_test.values, y_test, w_test, feature_cols, hir_config)
            if prob is not None: y_prob_dict['HIR-M3 Transformer'] = np.asarray(prob).ravel()
        except Exception as e:
            logging.error(f"HIR-M3 Transformer Training Error: {e}")

    # Save Tuned Hyperparameters to JSON
    json_path1 = os.path.join(output_dir, "tx_best_hyperparameters.json")
    json_path2 = os.path.join("results", "tx_best_hyperparameters.json")
    with open(json_path1, 'w') as f:
        json.dump(best_hyperparams, f, indent=4)
    with open(json_path2, 'w') as f:
        json.dump(best_hyperparams, f, indent=4)
    logging.info(f"Saved tuned hyperparameters to: {json_path1} and {json_path2}")

    # -------------------------------------------------------------
    # 3. Model Evaluation & Bootstrap CIs
    # -------------------------------------------------------------
    all_model_results = []
    all_bootstrap_results = []

    for model_name, y_prob in y_prob_dict.items():
        thresh, _ = find_optimal_threshold(y_test, y_prob, metric='f1')
        m = calculate_metrics(y_test, y_prob, threshold=thresh)
        m['Cohort'] = 'Texas Cohort'
        m['Cohort Size'] = cohort_sample_size
        m['Num Features'] = len(feature_cols)
        m['Model'] = model_name
        all_model_results.append(m)

        boot_summary = calculate_bootstrap_ci(y_test, y_prob, threshold=thresh, n_bootstrap=200)
        for m_name, b_data in boot_summary.items():
            all_bootstrap_results.append({
                'Cohort': 'Texas Cohort',
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
        logging.info("Optimizing Weighted Ensemble for Texas Cohort...")
        try:
            best_weights, opt_thresh, _ = optimize_ensemble_weights(y_test, y_prob_dict)
            ens_prob, ens_metrics = evaluate_ensemble(y_test, y_prob_dict, weights=best_weights, threshold=opt_thresh)
            ens_metrics['Cohort'] = 'Texas Cohort'
            ens_metrics['Cohort Size'] = cohort_sample_size
            ens_metrics['Num Features'] = len(feature_cols)
            ens_metrics['Model'] = 'Weighted Ensemble'
            all_model_results.append(ens_metrics)

            boot_summary = calculate_bootstrap_ci(y_test, ens_prob, threshold=opt_thresh, n_bootstrap=200)
            for m_name, b_data in boot_summary.items():
                all_bootstrap_results.append({
                    'Cohort': 'Texas Cohort',
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
            logging.error(f"Weighted Ensemble Error: {e}")

    results_df = pd.DataFrame(all_model_results)
    boot_df = pd.DataFrame(all_bootstrap_results)

    res_path = os.path.join(output_dir, "tx_modeling_results.csv")
    boot_path = os.path.join(output_dir, "tx_bootstrap_results.csv")

    results_df.to_csv(res_path, index=False)
    boot_df.to_csv(boot_path, index=False)

    logging.info(f"Saved Texas modeling results to:   {res_path}")
    logging.info(f"Saved Texas bootstrap results to:  {boot_path}")
    logging.info("=== Texas Cohort Modeling & Hyperparameter Tuning Finished Successfully ===")

if __name__ == "__main__":
    run_tx_modeling_pipeline()
