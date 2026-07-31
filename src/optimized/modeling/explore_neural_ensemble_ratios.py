import os
import sys
import time
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
from metrics import calculate_metrics, find_optimal_threshold

for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

COHORTS_FILES = {
    "Texas Cohort": "../data/processed_final_mergedDF_TX_condensed.csv"
}

def load_data(filepath):
    from run_neural_modeling import load_and_prep_data
    return load_and_prep_data(filepath)

def explore_neural_ensemble_ratios():
    logging.info("=================================================")
    logging.info("  PYTORCH NEURAL ENSEMBLE RATIO EXPLORATION")
    logging.info("=================================================")

    if not TORCH_AVAILABLE:
        logging.error(f"PyTorch not available ({TORCH_IMPORT_ERROR}). Cannot run neural ensemble ratio experiment.")
        return

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    filepath = list(COHORTS_FILES.values())[0]
    data = load_data(filepath)
    if data is None:
        logging.error("Failed to load dataset for neural ratio exploration.")
        return

    X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, sample_size = data

    if "TX" in filepath or "texas" in filepath.lower():
        logging.info("Using 100% of Texas dataset for neural ratio experiment (no 20% downsampling)...")
        w_train = np.ones_like(y_train, dtype=float)
        w_test = np.ones_like(y_test, dtype=float)
        sample_size = len(y_train) + len(y_test)
    else:
        from sklearn.model_selection import train_test_split
        logging.info("Sampling dataset by 20% (stratified) for neural ratio experiment...")
        X_train_scaled, _, y_train, _ = train_test_split(
            X_train_scaled, y_train, train_size=0.20, stratify=y_train, random_state=42
        )
        X_test_scaled, _, y_test, _ = train_test_split(
            X_test_scaled, y_test, train_size=0.20, stratify=y_test, random_state=42
        )
        w_train = np.ones_like(y_train, dtype=float)
        w_test = np.ones_like(y_test, dtype=float)
        sample_size = len(y_train) + len(y_test)

    logging.info(f"Dataset shape for neural ratio experiment: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}, Total={sample_size:,}")

    y_prob_dict = {}

    # PyTorch Neural Models
    logging.info("Training Standard MLP...")
    try:
        config_mlp = {'BATCH_SIZE': 256, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 10, 'DROPOUT': 0.2}
        prob = train_mlp(X_train_scaled.values, y_train, w_train, X_test_scaled.values, y_test, w_test, config_mlp)
        if prob is not None: y_prob_dict['Standard MLP'] = prob
    except Exception as e:
        logging.error(f"Standard MLP Error: {e}")

    # logging.info("Training Standard Transformer...")
    # try:
    #     config_trans = {'BATCH_SIZE': 64, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'HIDDEN_DIM': 64, 
    #                     'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.0, 'GAMMA': 0.0, 'PATIENCE': 3}
    #     prob = train_standard_transformer(X_train_scaled.values, y_train, w_train, X_test_scaled.values, y_test, w_test, feature_cols, config_trans)
    #     if prob is not None: y_prob_dict['Standard Transformer'] = prob
    # except Exception as e:
    #     logging.error(f"Standard Transformer Error: {e}")

    # 3. SAINT Transformer (COMMENTED OUT)
    # logging.info("Training SAINT Transformer...")
    # try:
    #     config_saint = {'BATCH_SIZE': 128, 'EMBED_DIM': 16, 'NUM_HEADS': 2, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 32, 
    #                     'LR': 1e-3, 'EPOCHS': 5, 'DROPOUT': 0.1}
    #     prob = train_saint(X_train_scaled.values, y_train, w_train, X_test_scaled.values, y_test, w_test, feature_cols, config_saint)
    #     if prob is not None: y_prob_dict['SAINT Transformer'] = prob
    # except Exception as e:
    #     logging.error(f"SAINT Transformer Error: {e}")

    logging.info("Training HIR-M3 Transformer...")
    try:
        config_hir = {'BATCH_SIZE': 64, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'HIDDEN_DIM': 64, 
                      'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.1, 'GAMMA': 0.5, 'PATIENCE': 3}
        prob = train_hir(X_train_scaled.values, y_train, w_train, X_test_scaled.values, y_test, w_test, feature_cols, config_hir)
        if prob is not None: y_prob_dict['HIR-M3 Transformer'] = prob
    except Exception as e:
        logging.error(f"HIR-M3 Transformer Error: {e}")

    hir_key = next((k for k in y_prob_dict.keys() if 'HIR' in k or 'Transformer' in k), None)
    if not hir_key:
        logging.error("HIR-M3 Transformer prediction is not available in trained models.")
        return

    y_prob_hir = y_prob_dict[hir_key]
    candidate_base_models = {k: v for k, v in y_prob_dict.items() if k != hir_key}

    if not candidate_base_models:
        logging.error("No base neural models available to blend with HIR-M3.")
        return

    ratios = [
        (0.10, 0.90, "10:90"),
        (0.20, 0.80, "20:80"),
        (0.30, 0.70, "30:70"),
        (0.40, 0.60, "40:60"),
        (0.50, 0.50, "50:50"),
        (0.60, 0.40, "60:40"),
        (0.70, 0.30, "70:30"),
        (0.80, 0.20, "80:20"),
        (0.90, 0.10, "90:10")
    ]

    all_ratio_results = []

    # Single standalone model baselines
    for model_name, y_prob in y_prob_dict.items():
        opt_th, _ = find_optimal_threshold(y_test, y_prob, metric='f1')
        m = calculate_metrics(y_test, y_prob, threshold=opt_th)
        m['Base Model'] = model_name
        m['Ratio Pair'] = "Standalone 100:0"
        m['Weight Base'] = 1.0
        m['Weight HIR'] = 0.0
        m['Optimal Threshold'] = opt_th
        m['Sample Size'] = sample_size
        all_ratio_results.append(m)

    logging.info("\n--- Evaluating Neural Ensemble Blending Ratios ---")
    for model_name, y_prob_base in candidate_base_models.items():
        for w_base, w_hir, ratio_str in ratios:
            blend_prob = w_base * y_prob_base + w_hir * y_prob_hir
            opt_th, _ = find_optimal_threshold(y_test, blend_prob, metric='f1')
            m = calculate_metrics(y_test, blend_prob, threshold=opt_th)
            m['Base Model'] = model_name
            m['Ratio Pair'] = f"{model_name} ({ratio_str}) HIR-M3"
            m['Weight Base'] = w_base
            m['Weight HIR'] = w_hir
            m['Optimal Threshold'] = opt_th
            m['Sample Size'] = sample_size
            all_ratio_results.append(m)

    ratio_df = pd.DataFrame(all_ratio_results)
    out_path = os.path.join(output_dir, "tx_neural_ensemble_ratio_exploration_results.csv")
    ratio_df.to_csv(out_path, index=False)
    logging.info(f"\nSaved neural ensemble ratio exploration results to {out_path}")
    logging.info("=== Neural Ratio Exploration Finished Successfully ===")

if __name__ == "__main__":
    explore_neural_ensemble_ratios()
