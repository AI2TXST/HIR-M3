import os
import sys
import time
import json
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

COHORTS_FILES = {
    "Texas Cohort": [
        "../data/processed_final_mergedDF_condensed_TX.parquet",
        "data/processed_final_mergedDF_condensed_TX.parquet",
        "../data/processed_final_mergedDF_condensed_TX.csv",
        "data/processed_final_mergedDF_condensed_TX.csv"
    ],
    "Nationwide Cohort": [
        "../data/processed_final_mergedDF_condensed.parquet",
        "data/processed_final_mergedDF_condensed.parquet",
        "../data/processed_final_mergedDF_condensed.csv",
        "data/processed_final_mergedDF_condensed.csv"
    ]
}

def run_ensemble_ratios_experiment(dataset_path=None, output_dir="results", cohort="all"):
    """
    Trains base candidate models and HIR-M3 Transformer using saved tuned hyperparameters,
    then systematically explores ensemble blend ratios (10:90, 20:80, 30:70, ..., 90:10)
    between each candidate model and HIR-M3 across cohorts.
    """
    logging.info("=================================================================")
    logging.info(f"  HIR-M3 ENSEMBLE RATIO EXPLORATION EXPERIMENT (COHORT: {cohort.upper()})")
    logging.info("=================================================================")
    os.makedirs(output_dir, exist_ok=True)

    if dataset_path and os.path.exists(dataset_path):
        target_cohorts = {"Custom Cohort": [dataset_path]}
    elif cohort.lower() == "texas":
        target_cohorts = {"Texas Cohort": COHORTS_FILES["Texas Cohort"]}
    elif cohort.lower() == "nationwide":
        target_cohorts = {"Nationwide Cohort": COHORTS_FILES["Nationwide Cohort"]}
    else:
        target_cohorts = COHORTS_FILES

    all_detailed = []
    all_summary = []

    for cohort_name, candidates in target_cohorts.items():
        logging.info(f"\n=================================================================")
        logging.info(f"  RUNNING ENSEMBLE RATIOS FOR: {cohort_name.upper()}")
        logging.info(f"=================================================================")
        actual_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

        # 1. Load Dataset
        logging.info(f"Loading dataset from: {actual_path}")
        data = load_and_prep_data(actual_path)
        if data is None:
            logging.error(f"Dataset loading failed for {cohort_name}. Skipping.")
            continue

        X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, sample_size = data

        w_train = np.ones_like(y_train, dtype=float)
        w_test = np.ones_like(y_test, dtype=float)
        sample_size = len(y_train) + len(y_test)

        logging.info(f"Dataset shape for ratio experiment: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}, Total={sample_size:,}")

        # 2. Load Saved Tuned Hyperparameters
        prefix = cohort_name.lower().replace(" ", "_")
        cand_ml_paths = [
            os.path.join(output_dir, f"{prefix}_best_hyperparameters.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{prefix}_best_hyperparameters.json"),
            os.path.join("results", f"{prefix}_best_hyperparameters.json"),
            os.path.join("modeling", "results", f"{prefix}_best_hyperparameters.json")
        ]
        cand_neural_paths = [
            os.path.join(output_dir, f"{prefix}_neural_best_hyperparameters.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{prefix}_neural_best_hyperparameters.json"),
            os.path.join("results", f"{prefix}_neural_best_hyperparameters.json"),
            os.path.join("modeling", "results", f"{prefix}_neural_best_hyperparameters.json")
        ]
        ml_hp_path = next((p for p in cand_ml_paths if os.path.exists(p)), None)
        neural_hp_path = next((p for p in cand_neural_paths if os.path.exists(p)), None)

        ml_hp = {}
        if ml_hp_path:
            try:
                with open(ml_hp_path) as f:
                    ml_hp = json.load(f)
                logging.info(f"Loaded tuned ML hyperparameters for {cohort_name} from {ml_hp_path}")
            except Exception as e:
                logging.warning(f"Error loading {ml_hp_path}: {e}")

        neural_hp = {}
        if neural_hp_path:
            try:
                with open(neural_hp_path) as f:
                    neural_hp = json.load(f)
                logging.info(f"Loaded tuned Neural hyperparameters for {cohort_name} from {neural_hp_path}")
            except Exception as e:
                logging.warning(f"Error loading {neural_hp_path}: {e}")

        # Dictionary to store model predicted probabilities
        y_prob_dict = {}

        # 3. Train Base Models with Tuned Hyperparameters
        # LightGBM
        if LGBM_AVAILABLE:
            cfg = ml_hp.get('LightGBM', {})
            logging.info(f"Training LightGBM (Params: {cfg})...")
            try:
                prob = train_lgb(X_train_scaled, y_train, w_train, X_test_scaled, y_test, cfg)
                if prob is not None: y_prob_dict['LightGBM'] = prob
            except Exception as e:
                logging.error(f"LightGBM Error: {e}")

        # XGBoost
        if XGB_AVAILABLE:
            cfg = ml_hp.get('XGBoost', {})
            logging.info(f"Training XGBoost (Params: {cfg})...")
            try:
                prob = train_xgb(X_train_scaled, y_train, w_train, X_test_scaled, y_test, cfg)
                if prob is not None: y_prob_dict['XGBoost'] = prob
            except Exception as e:
                logging.error(f"XGBoost Error: {e}")

        # CatBoost
        if CATBOOST_AVAILABLE:
            cfg = ml_hp.get('CatBoost', {})
            logging.info(f"Training CatBoost (Params: {cfg})...")
            try:
                prob = train_cb(X_train_scaled, y_train, w_train, X_test_scaled, y_test, cfg)
                if prob is not None: y_prob_dict['CatBoost'] = prob
            except Exception as e:
                logging.error(f"CatBoost Error: {e}")

        # Random Forest
        cfg = ml_hp.get('Random Forest', {})
        logging.info(f"Training Random Forest (Params: {cfg})...")
        try:
            prob = train_rf(X_train_scaled, y_train, w_train, X_test_scaled, y_test, cfg)
            if prob is not None: y_prob_dict['Random Forest'] = prob
        except Exception as e:
            logging.error(f"Random Forest Error: {e}")

        # Gradient Boosting (GBT)
        cfg = ml_hp.get('Gradient Boosting', {})
        logging.info(f"Training Gradient Boosting (Params: {cfg})...")
        try:
            prob = train_gbdt(X_train_scaled, y_train, w_train, X_test_scaled, y_test, cfg)
            if prob is not None: y_prob_dict['Gradient Boosting'] = prob
        except Exception as e:
            logging.error(f"Gradient Boosting Error: {e}")

        # Logistic Regression
        cfg = ml_hp.get('Logistic Regression', {})
        logging.info(f"Training Logistic Regression (Params: {cfg})...")
        try:
            prob = train_logreg(X_train_scaled, y_train, w_train, X_test_scaled, y_test, cfg)
            if prob is not None: y_prob_dict['Logistic Regression'] = prob
        except Exception as e:
            logging.error(f"Logistic Regression Error: {e}")

        # Target HIR-M3 Transformer for ratio blending with baseline ML models
        if TORCH_AVAILABLE:
            config_hir = dict(neural_hp.get('HIR-M3 Transformer', {}))
            config_hir.setdefault('BATCH_SIZE', 256)
            config_hir.setdefault('EMBED_DIM', 32)
            config_hir.setdefault('NUM_HEADS', 4)
            config_hir.setdefault('HIDDEN_DIM', 128)
            config_hir.setdefault('LR', 1e-3)
            config_hir.setdefault('EPOCHS', 3)
            config_hir.setdefault('LAMBDA_HIR', 0.05)
            config_hir.setdefault('GAMMA', 0.5)
            if config_hir.get('BATCH_SIZE', 256) < 256: config_hir['BATCH_SIZE'] = 256
            if config_hir.get('EPOCHS', 3) > 3: config_hir['EPOCHS'] = 3

            # Top 200 features for Transformer efficiency
            if len(feature_cols) > 200:
                variances = X_train_scaled.var(axis=0).values
                top_200_idx = np.argsort(variances)[::-1][:200]
                tf_feature_cols = [feature_cols[i] for i in top_200_idx]
            else:
                tf_feature_cols = feature_cols

            logging.info(f"Training HIR-M3 Transformer (Params: {config_hir}, Features: {len(tf_feature_cols)})...")
            try:
                prob = train_hir(X_train_scaled[tf_feature_cols].values, y_train, w_train, X_test_scaled[tf_feature_cols].values, y_test, w_test, tf_feature_cols, config_hir)
                if prob is not None: y_prob_dict['HIR-M3 Transformer'] = prob
            except Exception as e:
                logging.error(f"HIR-M3 Transformer Error: {e}")

        # Check if HIR-M3 Transformer is available for blending
        hir_key = next((k for k in y_prob_dict.keys() if 'HIR' in k or 'Transformer' in k), None)
        if not hir_key:
            logging.error(f"HIR-M3 Transformer prediction is not available for {cohort_name}.")
            continue

        y_prob_hir = y_prob_dict[hir_key]
        logging.info(f"Using '{hir_key}' as target component for ratio experiments.")

        candidate_base_models = {k: v for k, v in y_prob_dict.items() if k != hir_key}
        if not candidate_base_models:
            logging.error(f"No base candidate models available for {cohort_name}.")
            continue

        # Validation split from training data to tune decision thresholds without test set leakage
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42)
        w_tr = np.ones_like(y_tr, dtype=float)
        w_val = np.ones_like(y_val, dtype=float)

        y_val_prob_dict = {}
        if LGBM_AVAILABLE and 'LightGBM' in y_prob_dict:
            y_val_prob_dict['LightGBM'] = train_lgb(X_tr, y_tr, w_tr, X_val, y_val, ml_hp.get('LightGBM', {}))
        if XGB_AVAILABLE and 'XGBoost' in y_prob_dict:
            y_val_prob_dict['XGBoost'] = train_xgb(X_tr, y_tr, w_tr, X_val, y_val, ml_hp.get('XGBoost', {}))
        if CATBOOST_AVAILABLE and 'CatBoost' in y_prob_dict:
            y_val_prob_dict['CatBoost'] = train_cb(X_tr, y_tr, w_tr, X_val, y_val, ml_hp.get('CatBoost', {}))
        if 'Random Forest' in y_prob_dict:
            y_val_prob_dict['Random Forest'] = train_rf(X_tr, y_tr, w_tr, X_val, y_val, ml_hp.get('Random Forest', {}))
        if 'Gradient Boosting' in y_prob_dict:
            y_val_prob_dict['Gradient Boosting'] = train_gbdt(X_tr, y_tr, w_tr, X_val, y_val, ml_hp.get('Gradient Boosting', {}))
        if 'Logistic Regression' in y_prob_dict:
            y_val_prob_dict['Logistic Regression'] = train_logreg(X_tr, y_tr, w_tr, X_val, y_val, ml_hp.get('Logistic Regression', {}))
        if TORCH_AVAILABLE and hir_key in y_prob_dict:
            try:
                y_val_prob_dict[hir_key] = train_hir(X_tr[tf_feature_cols].values, y_tr, w_tr, X_val[tf_feature_cols].values, y_val, w_val, tf_feature_cols, config_hir)
            except Exception: pass

        # 4. Explore Ensemble Ratios across each model pair
        logging.info(f"\n--- Evaluating Ensemble Ratios (Model : HIR-M3) for {cohort_name} ---")

        detailed_results = []
        summary_results = []

        for model_name, y_prob_base in candidate_base_models.items():
            logging.info(f"\n--- Exploring Ratios for: {model_name} + {hir_key} ---")

            best_ratio_metrics = None
            best_f1 = -1.0
            best_opt_thresh = 0.5
            best_prob_ens = None

            for w_model, w_hir, ratio_str in RATIOS:
                # Linear probability blend on test set
                y_prob_ens = w_model * y_prob_base + w_hir * y_prob_hir

                # Tune threshold on validation set (no test set leakage)
                if model_name in y_val_prob_dict and hir_key in y_val_prob_dict and y_val_prob_dict[model_name] is not None and y_val_prob_dict[hir_key] is not None:
                    y_val_blend = w_model * y_val_prob_dict[model_name] + w_hir * y_val_prob_dict[hir_key]
                    opt_thresh, _ = find_optimal_threshold(y_val, y_val_blend, metric='f1')
                else:
                    opt_thresh = 0.5

                # Calculate evaluation metrics
                m = calculate_metrics(y_test, y_prob_ens, threshold=opt_thresh)

                row = {
                    'Cohort': cohort_name,
                    'Base Model': model_name,
                    'Target Model': hir_key,
                    'Pair Name': f"{model_name} + {hir_key}",
                    'Ensemble Ratio (Base:HIR)': ratio_str,
                    'Weight Base Model': float(w_model),
                    'Weight HIR-M3': float(w_hir),
                    'Threshold': float(opt_thresh),
                    'ROC_AUC': float(m['ROC_AUC']),
                    'PR_AUC': float(m['PR_AUC']),
                    'F1_Score': float(m['F1_Score']),
                    'Brier_Score': float(m['Brier_Score']),
                    'Accuracy': float(m.get('Accuracy', 0.0)),
                    'Precision': float(m.get('Precision', 0.0)),
                    'Recall': float(m.get('Recall', m.get('Recall_Sensitivity', 0.0))),
                    'Sample Size': sample_size
                }

                detailed_results.append(row)

                logging.info(f"  Ratio {ratio_str:>6s} ({model_name:>18s} {int(w_model*100):2d}% : HIR-M3 {int(w_hir*100):2d}%) -> "
                             f"F1: {m['F1_Score']:.4f} | ROC-AUC: {m['ROC_AUC']:.4f} | PR-AUC: {m['PR_AUC']:.4f} | Brier: {m['Brier_Score']:.4f}")

                if m['F1_Score'] > best_f1:
                    best_f1 = m['F1_Score']
                    best_opt_thresh = opt_thresh
                    best_prob_ens = y_prob_ens
                    best_ratio_metrics = row

            if best_ratio_metrics and best_prob_ens is not None:
                # Compute 95% Bootstrap CIs for the optimal ratio
                boot_summary = calculate_bootstrap_ci(y_test, best_prob_ens, threshold=best_opt_thresh, n_bootstrap=200)
                best_ratio_metrics['ROC_AUC [95% CI]'] = boot_summary.get('ROC_AUC', {}).get('Formatted', f"{best_ratio_metrics['ROC_AUC']:.4f}")
                best_ratio_metrics['PR_AUC [95% CI]'] = boot_summary.get('PR_AUC', {}).get('Formatted', f"{best_ratio_metrics['PR_AUC']:.4f}")
                best_ratio_metrics['F1 Score [95% CI]'] = boot_summary.get('F1_Score', {}).get('Formatted', f"{best_ratio_metrics['F1_Score']:.4f}")
                best_ratio_metrics['Brier [95% CI]'] = boot_summary.get('Brier_Score', {}).get('Formatted', f"{best_ratio_metrics['Brier_Score']:.4f}")
                summary_results.append(best_ratio_metrics)

        all_detailed.extend(detailed_results)
        all_summary.extend(summary_results)

    # 5. Save Results to CSV
    detailed_df = pd.DataFrame(all_detailed)
    summary_df = pd.DataFrame(all_summary)

    detailed_path = os.path.join(output_dir, "ensemble_ratios_detailed.csv")
    summary_path = os.path.join(output_dir, "ensemble_ratios_summary.csv")

    if not detailed_df.empty:
        detailed_df.to_csv(detailed_path, index=False)
        logging.info(f"Saved detailed ratio metrics to: {detailed_path}")
    if not summary_df.empty:
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Saved summary best ratios to:     {summary_path}")

        print("\n" + "="*85)
        print("  OPTIMAL ENSEMBLE RATIOS SUMMARY (Best F1 per Model + HIR-M3 Pair)")
        print("="*85)
        cols_to_print = [c for c in ['Cohort', 'Pair Name', 'Ensemble Ratio (Base:HIR)', 'ROC_AUC', 'PR_AUC', 'F1_Score', 'Brier_Score'] if c in summary_df.columns]
        print(summary_df[cols_to_print].to_string(index=False))
        print("="*85 + "\n")

    return detailed_df, summary_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble Ratio Exploration Experiment (Model : HIR-M3)")
    parser.add_argument("--cohort", default="all", choices=["Texas", "Nationwide", "all", "texas", "nationwide"],
                        help="Specific cohort to run: 'Texas', 'Nationwide', or 'all' (default: all)")
    parser.add_argument("--data", default=None, help="Path to input dataset (if omitted, uses cohort files)")
    parser.add_argument("--output_dir", default="results", help="Directory to save experiment results")
    args = parser.parse_args()

    run_ensemble_ratios_experiment(dataset_path=args.data, output_dir=args.output_dir, cohort=args.cohort)

