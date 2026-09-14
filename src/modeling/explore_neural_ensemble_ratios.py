import os
import sys
import time
import json
import gc
import logging
import pandas as pd
import numpy as np

# Add modeling directory to path
sys.path.append(os.path.dirname(__file__))

from models import (
    train_hir, train_mlp, train_standard_transformer, train_saint,
    TORCH_AVAILABLE, TORCH_IMPORT_ERROR
)
from metrics import calculate_metrics, find_optimal_threshold, calculate_bootstrap_ci

for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

COHORTS_FILES = {
    "Texas Cohort": [
        "../data/processed_final_mergedDF_condensed_TX.parquet",
        "data/processed_final_mergedDF_condensed_TX.parquet",
        "../data/processed_final_mergedDF_condensed_TX.csv",
        "data/processed_final_mergedDF_condensed_TX.csv",
    ],
    "Nationwide Cohort": [
        "../data/processed_final_mergedDF_condensed.parquet",
        "data/processed_final_mergedDF_condensed.parquet",
        "../data/processed_final_mergedDF_condensed.csv",
        "data/processed_final_mergedDF_condensed.csv",
    ]
}

def load_data(filepath):
    from run_neural_modeling import load_and_prep_data
    if isinstance(filepath, list):
        filepath = next((p for p in filepath if os.path.exists(p)), filepath[0])
    return load_and_prep_data(filepath)

def explore_neural_ensemble_ratios(cohort="all"):
    logging.info("=================================================================")
    logging.info(f"  PYTORCH NEURAL ENSEMBLE RATIO EXPLORATION (COHORT: {cohort.upper()})")
    logging.info("=================================================================")

    if not TORCH_AVAILABLE:
        logging.error(f"PyTorch not available ({TORCH_IMPORT_ERROR}). Cannot run neural ensemble ratio experiment.")
        return

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    if cohort.lower() == "texas":
        target_cohorts = {"Texas Cohort": COHORTS_FILES["Texas Cohort"]}
    elif cohort.lower() == "nationwide":
        target_cohorts = {"Nationwide Cohort": COHORTS_FILES["Nationwide Cohort"]}
    else:
        target_cohorts = COHORTS_FILES

    all_ratio_results = []
    all_summary_results = []

    for cohort_name, file_candidates in target_cohorts.items():
        logging.info(f"\n=================================================")
        logging.info(f"  PYTORCH NEURAL RATIO EXPLORATION: {cohort_name.upper()}")
        logging.info(f"=================================================")
        filepath = next((p for p in file_candidates if os.path.exists(p)), file_candidates[0])
        data = load_data(filepath)
        if data is None:
            logging.error(f"Failed to load dataset for {cohort_name}. Skipping.")
            continue

        X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, sample_size = data

        w_train = np.ones_like(y_train, dtype=float)
        w_test = np.ones_like(y_test, dtype=float)
        sample_size = len(y_train) + len(y_test)

        logging.info(f"Dataset shape for neural ratio experiment: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}, Total={sample_size:,}")

        # Load Saved Neural Hyperparameters
        prefix = cohort_name.lower().replace(" ", "_")
        cand_neural_paths = [
            os.path.join(output_dir, f"{prefix}_neural_best_hyperparameters.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{prefix}_neural_best_hyperparameters.json"),
            os.path.join("results", f"{prefix}_neural_best_hyperparameters.json"),
            os.path.join("modeling", "results", f"{prefix}_neural_best_hyperparameters.json")
        ]
        neural_hp_path = next((p for p in cand_neural_paths if os.path.exists(p)), None)
        neural_hp = {}
        if neural_hp_path:
            try:
                with open(neural_hp_path) as f:
                    neural_hp = json.load(f)
                logging.info(f"Loaded tuned Neural hyperparameters for {cohort_name} from {neural_hp_path}")
            except Exception as e:
                logging.warning(f"Error loading {neural_hp_path}: {e}")

        # Feature Subsetting for Tabular Transformers (OOM Memory Safety)
        if len(feature_cols) > 200:
            logging.info(f"Subsetting {len(feature_cols)} features to top 200 highest-variance features for Tabular Transformers...")
            variances = X_train_scaled.var(axis=0).values
            top_200_idx = np.argsort(variances)[::-1][:200]
            tf_feature_cols = [feature_cols[i] for i in top_200_idx]
            X_train_tf = X_train_scaled.iloc[:, top_200_idx]
            X_test_tf = X_test_scaled.iloc[:, top_200_idx]
        else:
            tf_feature_cols = feature_cols
            X_train_tf = X_train_scaled
            X_test_tf = X_test_scaled

        y_prob_dict = {}

        # 1. Standard MLP
        cfg_mlp = neural_hp.get('Standard MLP', {'BATCH_SIZE': 128, 'HIDDEN_DIM': 128, 'LR': 0.0005, 'EPOCHS': 10, 'DROPOUT': 0.2})
        logging.info(f"Training Standard MLP (Params: {cfg_mlp})...")
        try:
            prob = train_mlp(X_train_scaled.values, y_train, w_train, X_test_scaled.values, y_test, w_test, cfg_mlp)
            if prob is not None: y_prob_dict['Standard MLP'] = prob
        except Exception as e:
            logging.error(f"Standard MLP Error: {e}")

        # 2. Standard Tabular Transformer
        cfg_trans = neural_hp.get('Standard Transformer', {'BATCH_SIZE': 64, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 5, 'PATIENCE': 3})
        logging.info(f"Training Standard Transformer (Params: {cfg_trans})...")
        try:
            prob = train_standard_transformer(X_train_tf.values, y_train, w_train, X_test_tf.values, y_test, w_test, tf_feature_cols, cfg_trans)
            if prob is not None: y_prob_dict['Standard Transformer'] = prob
        except Exception as e:
            logging.error(f"Standard Transformer Error: {e}")

        # 3. SAINT Transformer
        cfg_saint = neural_hp.get('SAINT Transformer', {'BATCH_SIZE': 64, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 5, 'DROPOUT': 0.2})
        logging.info(f"Training SAINT Transformer (Params: {cfg_saint})...")
        try:
            prob = train_saint(X_train_tf.values, y_train, w_train, X_test_tf.values, y_test, w_test, tf_feature_cols, cfg_saint)
            if prob is not None: y_prob_dict['SAINT Transformer'] = prob
        except Exception as e:
            logging.error(f"SAINT Transformer Error: {e}")

        # 4. HIR-M3 Tabular Transformer (Target Component)
        cfg_hir = neural_hp.get('HIR-M3 Transformer', {'BATCH_SIZE': 64, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.05, 'GAMMA': 0.5, 'PATIENCE': 3})
        logging.info(f"Training HIR-M3 Transformer (Params: {cfg_hir})...")
        try:
            prob = train_hir(X_train_tf.values, y_train, w_train, X_test_tf.values, y_test, w_test, tf_feature_cols, cfg_hir)
            if prob is not None: y_prob_dict['HIR-M3 Transformer'] = prob
        except Exception as e:
            logging.error(f"HIR-M3 Transformer Error: {e}")

        hir_key = next((k for k in y_prob_dict.keys() if 'HIR' in k or 'Transformer' in k), None)
        if not hir_key:
            logging.error(f"HIR-M3 Transformer prediction is not available for {cohort_name}.")
            continue

        y_prob_hir = y_prob_dict[hir_key]
        candidate_base_models = {k: v for k, v in y_prob_dict.items() if k != hir_key}

        if not candidate_base_models:
            logging.error(f"No base neural models available for {cohort_name}.")
            continue

        ratios = [
            (1.00, 0.00, "100:0"),
            (0.90, 0.10, "90:10"),
            (0.80, 0.20, "80:20"),
            (0.70, 0.30, "70:30"),
            (0.60, 0.40, "60:40"),
            (0.50, 0.50, "50:50"),
            (0.40, 0.60, "40:60"),
            (0.30, 0.70, "30:70"),
            (0.20, 0.80, "20:80"),
            (0.10, 0.90, "10:90"),
            (0.00, 1.00, "0:100")
        ]

        logging.info(f"\n--- Evaluating Neural Ensemble Blending Ratios ({cohort_name}) ---")
        detailed_results = []
        summary_results = []

        for model_name, y_prob_base in candidate_base_models.items():
            best_f1 = -1.0
            best_ratio_row = None

            for w_base, w_hir, ratio_str in ratios:
                blend_prob = w_base * y_prob_base + w_hir * y_prob_hir
                opt_th, _ = find_optimal_threshold(y_test, blend_prob, metric='f1')
                m = calculate_metrics(y_test, blend_prob, threshold=opt_th)
                boot_summary = calculate_bootstrap_ci(y_test, blend_prob, threshold=opt_th, n_bootstrap=200)

                row = {
                    'Cohort': cohort_name,
                    'Base Model': model_name,
                    'Target Model': hir_key,
                    'Pair Name': f"{model_name} + {hir_key}",
                    'Ensemble Ratio (Base:HIR)': ratio_str,
                    'Weight Base': w_base,
                    'Weight HIR': w_hir,
                    'Threshold': float(opt_th),
                    'ROC_AUC': float(m['ROC_AUC']),
                    'ROC_AUC [95% CI]': boot_summary.get('ROC_AUC', {}).get('Formatted', f"{m['ROC_AUC']:.4f}"),
                    'PR_AUC': float(m['PR_AUC']),
                    'PR_AUC [95% CI]': boot_summary.get('PR_AUC', {}).get('Formatted', f"{m['PR_AUC']:.4f}"),
                    'F1_Score': float(m['F1_Score']),
                    'F1 Score [95% CI]': boot_summary.get('F1_Score', {}).get('Formatted', f"{m['F1_Score']:.4f}"),
                    'Brier_Score': float(m['Brier_Score']),
                    'Brier [95% CI]': boot_summary.get('Brier_Score', {}).get('Formatted', f"{m['Brier_Score']:.4f}"),
                    'Accuracy': float(m.get('Accuracy', 0.0)),
                    'Precision': float(m.get('Precision', 0.0)),
                    'Recall': float(m.get('Recall', m.get('Recall_Sensitivity', 0.0))),
                    'Sample Size': sample_size
                }
                detailed_results.append(row)

                logging.info(f"  Ratio {ratio_str:>6s} ({model_name:>22s} {int(w_base*100):2d}% : HIR-M3 {int(w_hir*100):2d}%) -> "
                             f"F1: {m['F1_Score']:.4f} | ROC-AUC: {m['ROC_AUC']:.4f} | PR-AUC: {m['PR_AUC']:.4f} | Brier: {m['Brier_Score']:.4f}")

                if m['F1_Score'] > best_f1:
                    best_f1 = m['F1_Score']
                    best_ratio_row = row

            if best_ratio_row:
                summary_results.append(best_ratio_row)

        all_ratio_results.extend(detailed_results)
        all_summary_results.extend(summary_results)

    if all_ratio_results:
        ratio_df = pd.DataFrame(all_ratio_results)
        summary_df = pd.DataFrame(all_summary_results)

        detailed_path = os.path.join(output_dir, "neural_ensemble_ratios_detailed.csv")
        summary_path = os.path.join(output_dir, "neural_ensemble_ratios_summary.csv")

        ratio_df.to_csv(detailed_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        logging.info(f"\nSaved detailed neural ratio metrics to: {detailed_path}")
        logging.info(f"Saved summary best neural ratios to:     {summary_path}")

        print("\n" + "="*85)
        print("  OPTIMAL NEURAL ENSEMBLE RATIOS SUMMARY (Best F1 per Neural Model + HIR-M3 Pair)")
        print("="*85)
        cols_to_print = [c for c in ['Cohort', 'Pair Name', 'Ensemble Ratio (Base:HIR)', 'ROC_AUC', 'PR_AUC', 'F1_Score', 'Brier_Score'] if c in summary_df.columns]
        print(summary_df[cols_to_print].to_string(index=False))
        print("="*85 + "\n")

    logging.info("=== Neural Ratio Exploration Finished Successfully ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Neural Ensemble Ratio Exploration")
    parser.add_argument("--cohort", default="all", choices=["Texas", "Nationwide", "all", "texas", "nationwide"],
                        help="Specific cohort to run: 'Texas', 'Nationwide', or 'all' (default: all)")
    args = parser.parse_args()

    explore_neural_ensemble_ratios(cohort=args.cohort)

