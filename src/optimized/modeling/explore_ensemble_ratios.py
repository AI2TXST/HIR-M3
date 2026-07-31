import os
import sys
import time
import logging
import gc
import numpy as np
import pandas as pd

# Add modeling directory to path
sys.path.append(os.path.dirname(__file__))

from run_modeling import load_and_prep_data
from models import (
    train_lgb, train_xgb, train_cb, train_rf, train_gbdt, train_logreg,
    train_mlp, train_standard_transformer, train_saint, train_hir,
    LGBM_AVAILABLE, XGB_AVAILABLE, CATBOOST_AVAILABLE, TORCH_AVAILABLE,
    LGBM_IMPORT_ERROR, XGB_IMPORT_ERROR, CATBOOST_IMPORT_ERROR, TORCH_IMPORT_ERROR
)
from metrics import calculate_metrics, find_optimal_threshold, calculate_bootstrap_ci

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensemble Split Ratios: (w_model, w_hir, ratio_str)
RATIOS = [
    (1.0, 0.0, "100:0"),
    (0.9, 0.1, "90:10"),
    (0.8, 0.2, "80:20"),
    (0.7, 0.3, "70:30"),
    (0.6, 0.4, "60:40"),
    (0.5, 0.5, "50:50"),
    (0.4, 0.6, "40:60"),
    (0.3, 0.7, "30:70"),
    (0.2, 0.8, "20:80"),
    (0.1, 0.9, "10:90"),
    (0.0, 1.0, "0:100"),
]

def run_ensemble_ratios_experiment(dataset_path="../data/processed_final_mergedDF_TX_condensed.csv", output_dir="results"):
    """
    Trains base candidate models and HIR-M3 Transformer, then systematically explores
    ensemble blend ratios (10:90, 20:80, 30:70, ..., 90:10) between each candidate model and HIR-M3.
    """
    logging.info("=================================================================")
    logging.info("  HIR-M3 ENSEMBLE RATIO EXPLORATION EXPERIMENT")
    logging.info("=================================================================")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Dataset
    logging.info(f"Loading dataset from: {dataset_path}")
    data = load_and_prep_data(dataset_path)
    if data is None:
        logging.error("Dataset loading failed. Aborting experiment.")
        return

    X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, sample_size = data

    if "TX" in dataset_path or "texas" in dataset_path.lower():
        logging.info("Using 100% of Texas dataset for ratio experiment (no 20% downsampling)...")
        w_train = np.ones_like(y_train, dtype=float)
        w_test = np.ones_like(y_test, dtype=float)
        sample_size = len(y_train) + len(y_test)
    else:
        from sklearn.model_selection import train_test_split
        logging.info("Sampling dataset by 20% (stratified) for ratio experiment...")
        X_train_scaled, _, y_train, _ = train_test_split(
            X_train_scaled, y_train, train_size=0.20, stratify=y_train, random_state=42
        )
        X_test_scaled, _, y_test, _ = train_test_split(
            X_test_scaled, y_test, train_size=0.20, stratify=y_test, random_state=42
        )
        w_train = np.ones_like(y_train, dtype=float)
        w_test = np.ones_like(y_test, dtype=float)
        sample_size = len(y_train) + len(y_test)

    logging.info(f"Dataset shape for ratio experiment: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}, Total={sample_size:,}")

    # Dictionary to store model predicted probabilities
    y_prob_dict = {}

    # 2. Train Base Models
    # LightGBM
    if LGBM_AVAILABLE:
        logging.info("Training LightGBM...")
        try:
            prob = train_lgb(X_train_scaled, y_train, w_train, X_test_scaled, y_test, {})
            if prob is not None: y_prob_dict['LightGBM'] = prob
        except Exception as e:
            logging.error(f"LightGBM Error: {e}")

    # XGBoost
    if XGB_AVAILABLE:
        logging.info("Training XGBoost...")
        try:
            prob = train_xgb(X_train_scaled, y_train, w_train, X_test_scaled, y_test, {})
            if prob is not None: y_prob_dict['XGBoost'] = prob
        except Exception as e:
            logging.error(f"XGBoost Error: {e}")

    # CatBoost
    if CATBOOST_AVAILABLE:
        logging.info("Training CatBoost...")
        try:
            prob = train_cb(X_train_scaled, y_train, w_train, X_test_scaled, y_test, {})
            if prob is not None: y_prob_dict['CatBoost'] = prob
        except Exception as e:
            logging.error(f"CatBoost Error: {e}")

    # Random Forest
    logging.info("Training Random Forest...")
    try:
        prob = train_rf(X_train_scaled, y_train, w_train, X_test_scaled, y_test, {})
        if prob is not None: y_prob_dict['Random Forest'] = prob
    except Exception as e:
        logging.error(f"Random Forest Error: {e}")

    # Gradient Boosting (GBT)
    logging.info("Training Gradient Boosting (GBT)...")
    try:
        prob = train_gbdt(X_train_scaled, y_train, w_train, X_test_scaled, y_test, {})
        if prob is not None: y_prob_dict['Gradient Boosting'] = prob
    except Exception as e:
        logging.error(f"Gradient Boosting Error: {e}")

    # Logistic Regression
    logging.info("Training Logistic Regression...")
    try:
        prob = train_logreg(X_train_scaled, y_train, w_train, X_test_scaled, y_test, {})
        if prob is not None: y_prob_dict['Logistic Regression'] = prob
    except Exception as e:
        logging.error(f"Logistic Regression Error: {e}")

    # Target HIR-M3 Transformer for ratio blending with baseline ML models
    if TORCH_AVAILABLE:
        logging.info("Training HIR-M3 Transformer...")
        try:
            config_hir = {'BATCH_SIZE': 64, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'HIDDEN_DIM': 64, 
                          'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.1, 'GAMMA': 0.5, 'PATIENCE': 3}
            prob = train_hir(X_train_scaled.values, y_train, w_train, X_test_scaled.values, y_test, w_test, feature_cols, config_hir)
            if prob is not None: y_prob_dict['HIR-M3 Transformer'] = prob
        except Exception as e:
            logging.error(f"HIR-M3 Transformer Error: {e}")

    # Check if HIR-M3 Transformer is available for blending
    hir_key = next((k for k in y_prob_dict.keys() if 'HIR' in k or 'Transformer' in k), None)
    if not hir_key:
        logging.error("HIR-M3 Transformer prediction is not available in trained models.")
        return

    y_prob_hir = y_prob_dict[hir_key]
    logging.info(f"Using '{hir_key}' as the target HIR-M3 component for ensemble ratio experiments.")

    # Select base models to pair with HIR-M3
    candidate_base_models = {k: v for k, v in y_prob_dict.items() if k != hir_key}
    if not candidate_base_models:
        logging.error("No base candidate models available to blend with HIR-M3.")
        return

    # 3. Explore Ensemble Ratios across each model pair
    logging.info("\n=================================================================")
    logging.info("  EVALUATING ENSEMBLE RATIOS (Model : HIR-M3)")
    logging.info("=================================================================")

    detailed_results = []
    summary_results = []

    for model_name, y_prob_base in candidate_base_models.items():
        logging.info(f"\n--- Exploring Ratios for: {model_name} + {hir_key} ---")

        best_ratio_metrics = None
        best_f1 = -1.0

        for w_model, w_hir, ratio_str in RATIOS:
            # Linear probability blend: P_blend = w_model * P_base + w_hir * P_hir
            y_prob_ens = w_model * y_prob_base + w_hir * y_prob_hir

            # Find optimal classification threshold
            opt_thresh, _ = find_optimal_threshold(y_test, y_prob_ens, metric='f1')

            # Calculate evaluation metrics
            m = calculate_metrics(y_test, y_prob_ens, threshold=opt_thresh)

            # Compute bootstrap 95% confidence intervals
            boot_summary = calculate_bootstrap_ci(y_test, y_prob_ens, threshold=opt_thresh, n_bootstrap=200)

            row = {
                'Base Model': model_name,
                'Target Model': hir_key,
                'Pair Name': f"{model_name} + {hir_key}",
                'Ensemble Ratio (Base:HIR)': ratio_str,
                'Weight Base Model': float(w_model),
                'Weight HIR-M3': float(w_hir),
                'Threshold': float(opt_thresh),
                'ROC_AUC': float(m['ROC_AUC']),
                'ROC_AUC [95% CI]': boot_summary.get('ROC_AUC', {}).get('Formatted', f"{m['ROC_AUC']:.4f}"),
                'PR_AUC': float(m['PR_AUC']),
                'PR_AUC [95% CI]': boot_summary.get('PR_AUC', {}).get('Formatted', f"{m['PR_AUC']:.4f}"),
                'F1_Score': float(m['F1_Score']),
                'F1 Score [95% CI]': boot_summary.get('F1_Score', {}).get('Formatted', f"{m['F1_Score']:.4f}"),
                'Brier_Score': float(m['Brier_Score']),
                'Brier [95% CI]': boot_summary.get('Brier_Score', {}).get('Formatted', f"{m['Brier_Score']:.4f}"),
                'Accuracy': float(m['Accuracy']),
                'Precision': float(m['Precision']),
                'Recall': float(m['Recall']),
                'Sample Size': sample_size
            }

            detailed_results.append(row)

            logging.info(f"  Ratio {ratio_str:>6s} ({model_name:>18s} {int(w_model*100):2d}% : HIR-M3 {int(w_hir*100):2d}%) -> "
                         f"F1: {m['F1_Score']:.4f} | ROC-AUC: {m['ROC_AUC']:.4f} | PR-AUC: {m['PR_AUC']:.4f} | Brier: {m['Brier_Score']:.4f}")

            if m['F1_Score'] > best_f1:
                best_f1 = m['F1_Score']
                best_ratio_metrics = row

        if best_ratio_metrics:
            summary_results.append(best_ratio_metrics)

    # 4. Save Results to CSV
    detailed_df = pd.DataFrame(detailed_results)
    summary_df = pd.DataFrame(summary_results)

    detailed_path = os.path.join(output_dir, "ensemble_ratios_detailed.csv")
    summary_path = os.path.join(output_dir, "ensemble_ratios_summary.csv")

    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    logging.info("\n=================================================================")
    logging.info(f"Saved detailed ratio metrics to: {detailed_path}")
    logging.info(f"Saved summary best ratios to:     {summary_path}")
    logging.info("=================================================================")

    # Print summary table of best ratios per model pair
    print("\n" + "="*85)
    print("  OPTIMAL ENSEMBLE RATIOS SUMMARY (Best F1 per Model + HIR-M3 Pair)")
    print("="*85)
    print(summary_df[['Pair Name', 'Ensemble Ratio (Base:HIR)', 'ROC_AUC', 'PR_AUC', 'F1_Score', 'Brier_Score']].to_string(index=False))
    print("="*85 + "\n")

    return detailed_df, summary_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble Ratio Exploration Experiment (Model : HIR-M3)")
    parser.add_argument("--data", default="../data/processed_final_mergedDF_TX_condensed.csv", help="Path to input dataset")
    parser.add_argument("--output_dir", default="results", help="Directory to save experiment results")
    args = parser.parse_args()

    run_ensemble_ratios_experiment(dataset_path=args.data, output_dir=args.output_dir)
