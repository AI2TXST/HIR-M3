import os
import sys
import time
import logging
import gc
import json
import numpy as np
import pandas as pd

# Add parent modeling directory to path for imports
MODELING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if MODELING_DIR not in sys.path:
    sys.path.insert(0, MODELING_DIR)

from models import (
    train_lgb, train_xgb, train_cb, train_rf, train_gbdt, train_logreg,
    train_mlp, train_hir,
    LGBM_AVAILABLE, XGB_AVAILABLE, CATBOOST_AVAILABLE, TORCH_AVAILABLE
)
from metrics import calculate_metrics, find_optimal_threshold
from run_tx_modeling import load_and_prep_tx_data
from run_tx_urban_rural import load_best_hyperparams

for h in logging.root.handlers[:]: 
    logging.root.removeHandler(h)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensemble Split Ratios: (w_model, w_hir, ratio_str)
RATIOS = [
    (0.9, 0.1, "90:10"),
    (0.8, 0.2, "80:20"),
    (0.7, 0.3, "70:30"),
    (0.6, 0.4, "60:40"),
    (0.5, 0.5, "50:50"),
    (0.4, 0.6, "40:60"),
    (0.3, 0.7, "30:70"),
    (0.2, 0.8, "20:80"),
    (0.1, 0.9, "10:90"),
    (1.0, 0.0, "100:0"),
    (0.0, 1.0, "0:100"),
]

def run_tx_ensemble_ratios_experiment(output_dir="../../results"):
    logging.info("=================================================================")
    logging.info("  TEXAS COHORT HIR-M3 ENSEMBLE RATIO EXPLORATION EXPERIMENT")
    logging.info("=================================================================")
    os.makedirs(output_dir, exist_ok=True)

    data = load_and_prep_tx_data()
    if data is None:
        logging.error("Texas dataset loading failed. Aborting experiment.")
        return

    X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, sample_size = data
    w_train = np.ones_like(y_train, dtype=float)
    w_test = np.ones_like(y_test, dtype=float)

    logging.info(f"Using 100% of Texas Dataset: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}, Total={sample_size:,}")

    best_hyperparams = load_best_hyperparams(output_dir=output_dir)
    y_prob_dict = {}

    # Train Baseline Models with Tuned Hyperparameters (Parallel)
    from concurrent.futures import ThreadPoolExecutor

    def _task_lgb():
        if not LGBM_AVAILABLE: return "LightGBM", None
        params = best_hyperparams.get('LightGBM', {})
        prob = train_lgb(X_train_scaled, y_train, w_train, X_test_scaled, y_test, params)
        return "LightGBM", np.asarray(prob).ravel() if prob is not None else None

    def _task_xgb():
        if not XGB_AVAILABLE: return "XGBoost", None
        params = best_hyperparams.get('XGBoost', {})
        prob = train_xgb(X_train_scaled, y_train, w_train, X_test_scaled, y_test, params)
        return "XGBoost", np.asarray(prob).ravel() if prob is not None else None

    def _task_cb():
        if not CATBOOST_AVAILABLE: return "CatBoost", None
        params = best_hyperparams.get('CatBoost', {})
        prob = train_cb(X_train_scaled, y_train, w_train, X_test_scaled, y_test, params)
        return "CatBoost", np.asarray(prob).ravel() if prob is not None else None

    def _task_rf():
        params = best_hyperparams.get('Random Forest', {})
        prob = train_rf(X_train_scaled, y_train, w_train, X_test_scaled, y_test, params)
        return "Random Forest", np.asarray(prob).ravel() if prob is not None else None

    def _task_gbdt():
        params = best_hyperparams.get('Gradient Boosting', {})
        prob = train_gbdt(X_train_scaled, y_train, w_train, X_test_scaled, y_test, params)
        return "Gradient Boosting", np.asarray(prob).ravel() if prob is not None else None

    def _task_logreg():
        params = best_hyperparams.get('Logistic Regression', {})
        prob = train_logreg(X_train_scaled, y_train, w_train, X_test_scaled, y_test, params)
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

    # Train PyTorch Neural Models with Tuned Hyperparameters
    if TORCH_AVAILABLE:
        logging.info("Training Standard MLP Baseline...")
        try:
            cfg_mlp = best_hyperparams.get('Standard MLP', {'BATCH_SIZE': 256, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 10, 'DROPOUT': 0.2})
            prob = train_mlp(X_train_scaled.values, y_train, w_train, X_test_scaled.values, y_test, w_test, cfg_mlp)
            if prob is not None: y_prob_dict['Standard MLP'] = np.asarray(prob).ravel()
        except Exception as e:
            logging.error(f"Standard MLP Error: {e}")

        logging.info("Training HIR-M3 Transformer...")
        try:
            cfg_hir = best_hyperparams.get('HIR-M3 Transformer', {'BATCH_SIZE': 128, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 64, 'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.1, 'GAMMA': 0.5, 'PATIENCE': 3})
            prob = train_hir(X_train_scaled.values, y_train, w_train, X_test_scaled.values, y_test, w_test, feature_cols, cfg_hir)
            if prob is not None: y_prob_dict['HIR-M3 Transformer'] = np.asarray(prob).ravel()
        except Exception as e:
            logging.error(f"HIR-M3 Transformer Error: {e}")

    hir_key = next((k for k in y_prob_dict.keys() if 'HIR' in k or 'Transformer' in k), None)
    if not hir_key:
        logging.error("HIR-M3 Transformer prediction is not available in trained models.")
        return

    y_prob_hir = y_prob_dict[hir_key]
    candidate_base_models = {k: v for k, v in y_prob_dict.items() if k != hir_key}

    detailed_results = []
    summary_results = []

    logging.info("\n--- Evaluating Blending Ratios against HIR-M3 ---")
    for model_name, y_prob_base in candidate_base_models.items():
        pair_name = f"{model_name} + {hir_key}"
        best_f1_for_pair = -1.0
        best_ratio_metrics = None

        for w_base, w_hir, ratio_str in RATIOS:
            blend_prob = w_base * y_prob_base + w_hir * y_prob_hir
            opt_thresh, _ = find_optimal_threshold(y_test, blend_prob, metric='f1')
            metrics = calculate_metrics(y_test, blend_prob, threshold=opt_thresh)

            metrics['Pair Name'] = pair_name
            metrics['Base Model'] = model_name
            metrics['HIR Model'] = hir_key
            metrics['Weight Base'] = w_base
            metrics['Weight HIR'] = w_hir
            metrics['Ensemble Ratio (Base:HIR)'] = ratio_str
            metrics['Optimal Threshold'] = opt_thresh
            metrics['Sample Size'] = sample_size

            detailed_results.append(metrics)

            if metrics['F1_Score'] > best_f1_for_pair:
                best_f1_for_pair = metrics['F1_Score']
                best_ratio_metrics = metrics

        if best_ratio_metrics:
            summary_results.append(best_ratio_metrics)

    detailed_df = pd.DataFrame(detailed_results)
    summary_df = pd.DataFrame(summary_results)

    detailed_path = os.path.join(output_dir, "tx_ensemble_ratios_detailed.csv")
    summary_path = os.path.join(output_dir, "tx_ensemble_ratios_summary.csv")

    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    logging.info(f"Saved detailed ratio metrics to: {detailed_path}")
    logging.info(f"Saved summary best ratios to:     {summary_path}")
    logging.info("=== Texas Ensemble Ratio Experiment Finished Successfully ===")

if __name__ == "__main__":
    run_tx_ensemble_ratios_experiment()
