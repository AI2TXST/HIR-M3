import os
import sys
import time
import logging
import json
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add modeling directory to path
sys.path.append(os.path.dirname(__file__))

from models import (
    train_mlp, train_standard_transformer, train_saint, train_hir,
    TORCH_AVAILABLE, TORCH_IMPORT_ERROR
)
from metrics import calculate_metrics, calculate_bootstrap_ci, find_optimal_threshold
from optimize_ensemble import optimize_ensemble_weights, evaluate_ensemble

for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

COHORTS_FILES = {
    "Nationwide Cohort": [
        "../data/processed_final_mergedDF_condensed.parquet",
        "data/processed_final_mergedDF_condensed.parquet",
        "../data/processed_final_mergedDF_condensed.csv",
        "data/processed_final_mergedDF_condensed.csv"
    ],
    "Texas Cohort": [
        "../data/processed_final_mergedDF_condensed_TX.parquet",
        "data/processed_final_mergedDF_condensed_TX.parquet",
        "../data/processed_final_mergedDF_condensed_TX.csv",
        "data/processed_final_mergedDF_condensed_TX.csv"
    ]
}

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
                "../data/processed_final_mergedDF_TX.parquet",
                "data/processed_final_mergedDF_TX.parquet",
                "../data/processed_final_mergedDF_TX.csv",
                "data/processed_final_mergedDF_TX.csv"
            ]
        else:
            candidates = [
                "../data/processed_final_mergedDF.parquet",
                "data/processed_final_mergedDF.parquet",
                "../data/processed_final_mergedDF.csv",
                "data/processed_final_mergedDF.csv"
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
    logging.info(f"Loaded neural modeling dataset shape: {df.shape} (from {initial_total_rows:,} raw rows)")

    drop_cols = [
        'BENE_ID', 'Beneficiary_ID', 'Assessment_Effective_Date', 'COUNTYFIPS',
        'Days_Cared_For', 'ever_deceased', 'NumVisits', 'DaysBetweenVisits',
        'PrevVisitDate', 'Last_Assessment_Date', t_col
    ]
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

# ---------------------------------------------------------
# HYPERPARAMETER TUNING SEARCH ROUTINES FOR ALL 4 NEURAL MODELS
# ---------------------------------------------------------
def tune_mlp(X_train, y_train, w_train, X_val, y_val, w_val):
    logging.info("--- Hyperparameter Tuning for Standard MLP ---")
    grid = [
        {'BATCH_SIZE': 256, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 10, 'DROPOUT': 0.1},
        {'BATCH_SIZE': 256, 'HIDDEN_DIM': 256, 'LR': 1e-3, 'EPOCHS': 10, 'DROPOUT': 0.2},
        {'BATCH_SIZE': 128, 'HIDDEN_DIM': 128, 'LR': 5e-4, 'EPOCHS': 10, 'DROPOUT': 0.2},
    ]
    best_f1 = -1.0
    best_cfg = grid[0]
    for cfg in grid:
        try:
            prob = train_mlp(X_train, y_train, w_train, X_val, y_val, w_val, cfg)
            if prob is not None:
                opt_th, _ = find_optimal_threshold(y_val, prob, metric='f1')
                m = calculate_metrics(y_val, prob, threshold=opt_th)
                if m['F1_Score'] > best_f1:
                    best_f1 = m['F1_Score']
                    best_cfg = cfg
        except Exception as e:
            logging.error(f"MLP Tuning Error ({cfg}): {e}")
    logging.info(f"[Standard MLP] Best Parameters: {best_cfg} (Validation F1: {best_f1:.4f})")
    return best_cfg

def tune_standard_transformer(X_train, y_train, w_train, X_val, y_val, w_val, feature_cols):
    logging.info("--- Hyperparameter Tuning for Standard Tabular Transformer ---")
    grid = [
        {'BATCH_SIZE': 128, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 64, 'LR': 1e-3, 'EPOCHS': 5, 'PATIENCE': 3},
        {'BATCH_SIZE': 64, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 5, 'PATIENCE': 3},
    ]
    best_f1 = -1.0
    best_cfg = grid[0]
    for cfg in grid:
        try:
            prob = train_standard_transformer(X_train, y_train, w_train, X_val, y_val, w_val, feature_cols, cfg)
            if prob is not None:
                opt_th, _ = find_optimal_threshold(y_val, prob, metric='f1')
                m = calculate_metrics(y_val, prob, threshold=opt_th)
                if m['F1_Score'] > best_f1:
                    best_f1 = m['F1_Score']
                    best_cfg = cfg
        except Exception as e:
            logging.error(f"Standard Transformer Tuning Error ({cfg}): {e}")
    logging.info(f"[Standard Transformer] Best Parameters: {best_cfg} (Validation F1: {best_f1:.4f})")
    return best_cfg

def tune_saint(X_train, y_train, w_train, X_val, y_val, w_val, feature_cols):
    logging.info("--- Hyperparameter Tuning for SAINT Transformer ---")
    grid = [
        {'BATCH_SIZE': 128, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 64, 'LR': 1e-3, 'EPOCHS': 5, 'DROPOUT': 0.1},
        {'BATCH_SIZE': 64, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 5, 'DROPOUT': 0.2},
    ]
    best_f1 = -1.0
    best_cfg = grid[0]
    for cfg in grid:
        try:
            prob = train_saint(X_train, y_train, w_train, X_val, y_val, w_val, feature_cols, cfg)
            if prob is not None:
                opt_th, _ = find_optimal_threshold(y_val, prob, metric='f1')
                m = calculate_metrics(y_val, prob, threshold=opt_th)
                if m['F1_Score'] > best_f1:
                    best_f1 = m['F1_Score']
                    best_cfg = cfg
        except Exception as e:
            logging.error(f"SAINT Tuning Error ({cfg}): {e}")
    logging.info(f"[SAINT Transformer] Best Parameters: {best_cfg} (Validation F1: {best_f1:.4f})")
    return best_cfg

def tune_hir(X_train, y_train, w_train, X_val, y_val, w_val, feature_cols):
    logging.info("--- Hyperparameter Tuning for HIR-M3 Tabular Transformer ---")
    grid = [
        {'BATCH_SIZE': 128, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 64, 'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.1, 'GAMMA': 0.5, 'PATIENCE': 3},
        {'BATCH_SIZE': 64, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.05, 'GAMMA': 0.5, 'PATIENCE': 3},
    ]
    best_f1 = -1.0
    best_cfg = grid[0]
    for cfg in grid:
        try:
            prob = train_hir(X_train, y_train, w_train, X_val, y_val, w_val, feature_cols, cfg)
            if prob is not None:
                opt_th, _ = find_optimal_threshold(y_val, prob, metric='f1')
                m = calculate_metrics(y_val, prob, threshold=opt_th)
                if m['F1_Score'] > best_f1:
                    best_f1 = m['F1_Score']
                    best_cfg = cfg
        except Exception as e:
            logging.error(f"HIR Tuning Error ({cfg}): {e}")
    logging.info(f"[HIR-M3 Transformer] Best Parameters: {best_cfg} (Validation F1: {best_f1:.4f})")
    return best_cfg

def run_neural_pipeline(selected_cohort="all"):
    logging.info("=================================================================")
    logging.info("  STARTING PYTORCH NEURAL & TRANSFORMER PIPELINE (ALL 4 MODELS)")
    logging.info("=================================================================")
    logging.info(f"PyTorch Available: {TORCH_AVAILABLE}")

    if not TORCH_AVAILABLE:
        logging.error(f"PyTorch is not available: {TORCH_IMPORT_ERROR}. Aborting pipeline.")
        return

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
        logging.info(f"\n--- Running Neural Models for Cohort: {cohort_name} ---")
        data = load_and_prep_data(filepath)
        if data is None:
            continue

        X_train, X_test, y_train, y_test, feature_cols, cohort_sample_size = data
        w_train = np.ones_like(y_train, dtype=float)
        w_test = np.ones_like(y_test, dtype=float)

        # Validation split for hyperparameter search
        X_tr_val, X_val, y_tr_val, y_val = train_test_split(
            X_train.values, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        w_tr_val = np.ones_like(y_tr_val, dtype=float)
        w_val = np.ones_like(y_val, dtype=float)

        best_hyperparams = {}
        y_prob_dict = {}

        # -------------------------------------------------------------
        # SEQUENTIAL STABLE EXECUTION OF ALL 4 PYTORCH NEURAL MODELS
        # -------------------------------------------------------------
        # 1. Standard MLP (Pre-tuned best parameters on full feature set)
        best_mlp_cfg = {'BATCH_SIZE': 128, 'HIDDEN_DIM': 128, 'LR': 0.0005, 'EPOCHS': 10, 'DROPOUT': 0.2}
        logging.info(f"Using pre-set best parameters for Standard MLP: {best_mlp_cfg}")
        try:
            cfg = best_mlp_cfg
            best_hyperparams['Standard MLP'] = cfg
            prob = train_mlp(X_train.values, y_train, w_train, X_test.values, y_test, w_test, cfg)
            if prob is not None: y_prob_dict['Standard MLP'] = prob
        except Exception as e:
            logging.error(f"Standard MLP Error: {e}")

        # Feature Subsetting for Transformer Models (O(N^2) memory safety)
        if len(feature_cols) > 200:
            logging.info(f"Subsetting {len(feature_cols)} features to top 200 highest-variance features for Tabular Transformers to prevent OOM...")
            variances = X_train.var(axis=0).values
            top_200_idx = np.argsort(variances)[::-1][:200]
            tf_feature_cols = [feature_cols[i] for i in top_200_idx]
            
            X_train_tf = X_train.iloc[:, top_200_idx]
            X_test_tf = X_test.iloc[:, top_200_idx]
            X_tr_val_tf = X_tr_val[:, top_200_idx]
            X_val_tf = X_val[:, top_200_idx]
        else:
            tf_feature_cols = feature_cols
            X_train_tf = X_train
            X_test_tf = X_test
            X_tr_val_tf = X_tr_val
            X_val_tf = X_val

        # 2. Standard Tabular Transformer
        logging.info("Starting Standard Tabular Transformer tuning & training...")
        try:
            cfg = tune_standard_transformer(X_tr_val_tf, y_tr_val, w_tr_val, X_val_tf, y_val, w_val, tf_feature_cols)
            best_hyperparams['Standard Transformer'] = cfg
            prob = train_standard_transformer(X_train_tf.values, y_train, w_train, X_test_tf.values, y_test, w_test, tf_feature_cols, cfg)
            if prob is not None: y_prob_dict['Standard Transformer'] = prob
        except Exception as e:
            logging.error(f"Standard Transformer Error: {e}")

        # 3. SAINT Transformer
        logging.info("Starting SAINT Transformer tuning & training...")
        try:
            cfg = tune_saint(X_tr_val_tf, y_tr_val, w_tr_val, X_val_tf, y_val, w_val, tf_feature_cols)
            best_hyperparams['SAINT Transformer'] = cfg
            prob = train_saint(X_train_tf.values, y_train, w_train, X_test_tf.values, y_test, w_test, tf_feature_cols, cfg)
            if prob is not None: y_prob_dict['SAINT Transformer'] = prob
        except Exception as e:
            logging.error(f"SAINT Transformer Error: {e}")

        # 4. HIR-M3 Tabular Transformer
        logging.info("Starting HIR-M3 Tabular Transformer tuning & training...")
        try:
            cfg = tune_hir(X_tr_val_tf, y_tr_val, w_tr_val, X_val_tf, y_val, w_val, tf_feature_cols)
            best_hyperparams['HIR-M3 Transformer'] = cfg
            prob = train_hir(X_train_tf.values, y_train, w_train, X_test_tf.values, y_test, w_test, tf_feature_cols, cfg)
            if prob is not None: y_prob_dict['HIR-M3 Transformer'] = prob
        except Exception as e:
            logging.error(f"HIR-M3 Transformer Error: {e}")

        # Save best neural hyperparameters to JSON
        prefix = cohort_name.lower().replace(" ", "_")
        hp_save_path = os.path.join(output_dir, f"{prefix}_neural_best_hyperparameters.json")
        try:
            import json
            with open(hp_save_path, 'w') as f:
                json.dump(best_hyperparams, f, indent=4)
            logging.info(f"Saved tuned neural hyperparameters for {cohort_name} to {hp_save_path}")
        except Exception as e:
            logging.warning(f"Failed to save neural hyperparameters to {hp_save_path}: {e}")

        # Predict validation probabilities for threshold tuning (no test set leakage)
        y_val_prob_dict = {}
        try:
            y_val_prob_dict['Standard MLP'] = train_mlp(X_tr_val.values, y_tr_val, w_tr_val, X_val.values, y_val, w_val, best_mlp_cfg)
        except Exception: pass

        if 'Standard Transformer' in best_hyperparams:
            try:
                y_val_prob_dict['Standard Transformer'] = train_standard_transformer(X_tr_val_tf.values, y_tr_val, w_tr_val, X_val_tf.values, y_val, w_val, tf_feature_cols, best_hyperparams['Standard Transformer'])
            except Exception: pass

        if 'SAINT Transformer' in best_hyperparams:
            try:
                y_val_prob_dict['SAINT Transformer'] = train_saint(X_tr_val_tf.values, y_tr_val, w_tr_val, X_val_tf.values, y_val, w_val, tf_feature_cols, best_hyperparams['SAINT Transformer'])
            except Exception: pass

        if 'HIR-M3 Transformer' in best_hyperparams:
            try:
                y_val_prob_dict['HIR-M3 Transformer'] = train_hir(X_tr_val_tf.values, y_tr_val, w_tr_val, X_val_tf.values, y_val, w_val, tf_feature_cols, best_hyperparams['HIR-M3 Transformer'])
            except Exception: pass

        # Evaluate individual neural models on held-out test set
        model_results_cohort = []
        boot_results_cohort = []
        for model_name, y_prob in y_prob_dict.items():
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



        gc.collect()

    if all_model_results:
        results_df = pd.DataFrame(all_model_results)
        results_path = os.path.join(output_dir, "neural_modeling_results_all.csv")
        results_df.to_csv(results_path, index=False)
        logging.info(f"\nSaved consolidated neural modeling results to {results_path}")

    if all_bootstrap_results:
        boot_df = pd.DataFrame(all_bootstrap_results)
        boot_path = os.path.join(output_dir, "neural_bootstrap_results_all.csv")
        boot_df.to_csv(boot_path, index=False)
        logging.info(f"Saved consolidated neural bootstrap results to {boot_path}")

    logging.info("=== Neural Modeling Pipeline Finished Successfully ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Neural Modeling Pipeline")
    parser.add_argument("--cohort", choices=["Texas", "Nationwide", "all"], default="all",
                        help="Specify cohort to train (Texas, Nationwide, or all)")
    args = parser.parse_args()
    run_neural_pipeline(args.cohort)
