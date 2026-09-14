import os
import sys
import time
import json
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
RUCA_COL = "RUCA_Category"

def load_and_prep_urban_rural_data(filepath, area_type="urban", target_col=TARGET_COL):
    """
    Loads dataset in memory-efficient chunks, filters for Urban vs Rural, 
    samples to 20% for nationwide, and scales features for PyTorch Neural Models.
    """
    actual_path = None
    if isinstance(filepath, (list, tuple)):
        actual_path = next((p for p in filepath if os.path.exists(p)), None)
    elif os.path.exists(str(filepath)):
        actual_path = str(filepath)

    if not actual_path:
        logging.error(f"File not found for path: {filepath}")
        return None

    filepath = actual_path

    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=50000, low_memory=False):
            if RUCA_COL in chunk.columns:
                if area_type == "urban":
                    chunk_sub = chunk[chunk[RUCA_COL].astype(str).str.lower() == "urban"]
                else:
                    chunk_sub = chunk[chunk[RUCA_COL].astype(str).str.lower() == "rural"]
                chunks.append(chunk_sub)
            else:
                chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

    if RUCA_COL in df.columns:
        if area_type == "urban":
            df = df[df[RUCA_COL].astype(str).str.lower() == "urban"]
        else:
            df = df[df[RUCA_COL].astype(str).str.lower() == "rural"]

    logging.info(f"Loaded {len(df):,} rows for {area_type.upper()} setting from {os.path.basename(filepath)}")

    if target_col not in df.columns:
        possible_targets = [c for c in df.columns if 'target' in c.lower() or 'readmit' in c.lower()]
        if possible_targets:
            target_col = possible_targets[0]
        else:
            logging.error(f"Target column '{target_col}' not found.")
            return None

    # Memory Downcasting
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    # 50% Stratified Sample for Nationwide Cohort; keep Texas at 100%
    if "TX" not in str(filepath) and "texas" not in str(filepath).lower():
        if len(df) > 50000:
            logging.info(f"Sampling 50% of {len(df):,} rows for Nationwide Neural Urban/Rural execution...")
            df, _ = train_test_split(df, train_size=0.50, stratify=df[target_col], random_state=42)
            logging.info(f"Sampled Nationwide dataset size: {len(df):,} rows")

    # Drop non-feature columns, Beneficiary ID variants, and non-numeric object/string columns
    drop_cols = [
        target_col, 'ID', 'Patient_ID', 'ZipCode', RUCA_COL, 'Agency_Medicare_Number',
        'Facility_Internal_ID', 'COUNTY_NAME', 'BENE_ID', 'Beneficiary_ID',
        'Assessment_Effective_Date', 'COUNTYFIPS', 'Days_Cared_For', 'ever_deceased',
        'NumVisits', 'DaysBetweenVisits', 'PrevVisitDate', 'Last_Assessment_Date'
    ]
    id_cols = [c for c in df.columns if 'beneficiary' in c.lower() or 'bene_id' in c.lower() or c.lower().endswith('id')]
    non_numeric = df.select_dtypes(include=['object', 'string', 'category']).columns
    drop_cols = list(set(drop_cols).union(set(id_cols)).union(set(non_numeric)))

    feature_cols = [c for c in df.columns if c not in drop_cols and not c.startswith('Unnamed')]

    # Coerce features to numeric float32, dropping uncoercible string columns
    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        if X_df[col].dtype == object or X_df[col].dtype == str or X_df[col].dtype.name == 'category':
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

    X_df = X_df.dropna(how='all', axis=1)
    X_df = X_df.fillna(X_df.median()).astype(np.float32)
    feature_cols = list(X_df.columns)

    X = X_df
    y = df[target_col].values.astype(np.int8)

    # Train / Test Split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train).astype(np.float32), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test).astype(np.float32), columns=feature_cols, index=X_test.index)

    cohort_sample_size = len(df)

    del df, X, y, X_train, X_test
    gc.collect()

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, cohort_sample_size


def run_urban_rural_neural_pipeline(selected_cohort="all"):
    logging.info("=========================================================================")
    logging.info("  STARTING PYTORCH NEURAL URBAN vs. RURAL PIPELINE (ALL 4 NEURAL MODELS)")
    logging.info("=========================================================================")
    logging.info(f"PyTorch Available: {TORCH_AVAILABLE}")

    if not TORCH_AVAILABLE:
        logging.error(f"PyTorch not available ({TORCH_IMPORT_ERROR}). Aborting pipeline.")
        return

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    all_model_results = []
    all_bootstrap_results = []

    if selected_cohort.lower().startswith("tx") or selected_cohort.lower().startswith("texas"):
        target_files = {k: v for k, v in COHORTS_FILES.items() if "Texas" in k}
    elif selected_cohort.lower().startswith("nat") or selected_cohort.lower().startswith("nationwide"):
        target_files = {k: v for k, v in COHORTS_FILES.items() if "Nationwide" in k}
    else:
        target_files = COHORTS_FILES

    for cohort_name, filepath in target_files.items():
        # Load saved tuned neural hyperparameters if available
        prefix = cohort_name.lower().replace(" ", "_")
        candidate_neural_paths = [
            os.path.join(output_dir, f"{prefix}_neural_best_hyperparameters.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{prefix}_neural_best_hyperparameters.json"),
            os.path.join("results", f"{prefix}_neural_best_hyperparameters.json"),
            os.path.join("modeling", "results", f"{prefix}_neural_best_hyperparameters.json")
        ]
        neural_hp_path = next((p for p in candidate_neural_paths if os.path.exists(p)), None)
        neural_hp = {}
        if neural_hp_path:
            try:
                with open(neural_hp_path) as f:
                    neural_hp = json.load(f)
                logging.info(f"Loaded tuned Neural hyperparameters for {cohort_name} from {neural_hp_path}")
            except Exception as e:
                logging.warning(f"Error loading {neural_hp_path}: {e}")

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

            # Feature Subsetting for Tabular Transformers (OOM Memory Safety)
            if len(feature_cols) > 200:
                logging.info(f"Subsetting {len(feature_cols)} features to top 200 highest-variance features for Tabular Transformers...")
                variances = X_train.var(axis=0).values
                top_200_idx = np.argsort(variances)[::-1][:200]
                tf_feature_cols = [feature_cols[i] for i in top_200_idx]
                X_train_tf = X_train.iloc[:, top_200_idx]
                X_test_tf = X_test.iloc[:, top_200_idx]
            else:
                tf_feature_cols = feature_cols
                X_train_tf = X_train
                X_test_tf = X_test

            y_prob_dict = {}

            # 1. Standard MLP Baseline
            logging.info("Training Standard MLP Baseline...")
            try:
                config_mlp = neural_hp.get('Standard MLP', {'BATCH_SIZE': 128, 'HIDDEN_DIM': 128, 'LR': 0.0005, 'EPOCHS': 10, 'DROPOUT': 0.2})
                prob = train_mlp(X_train.values, y_train, w_train, X_test.values, y_test, w_test, config_mlp)
                if prob is not None: y_prob_dict['Standard MLP'] = prob
            except Exception as e:
                logging.error(f"Standard MLP Error: {e}")

            # 2. Standard Tabular Transformer
            logging.info("Training Standard Tabular Transformer...")
            try:
                config_trans = neural_hp.get('Standard Transformer', {'BATCH_SIZE': 64, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 5, 'PATIENCE': 3})
                prob = train_standard_transformer(X_train_tf.values, y_train, w_train, X_test_tf.values, y_test, w_test, tf_feature_cols, config_trans)
                if prob is not None: y_prob_dict['Standard Transformer'] = prob
            except Exception as e:
                logging.error(f"Standard Transformer Error: {e}")

            # 3. SAINT Transformer
            logging.info("Training SAINT Transformer...")
            try:
                config_saint = neural_hp.get('SAINT Transformer', {'BATCH_SIZE': 64, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'NUM_LAYERS': 1, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 5, 'DROPOUT': 0.2})
                prob = train_saint(X_train_tf.values, y_train, w_train, X_test_tf.values, y_test, w_test, tf_feature_cols, config_saint)
                if prob is not None: y_prob_dict['SAINT Transformer'] = prob
            except Exception as e:
                logging.error(f"SAINT Transformer Error: {e}")

            # 4. HIR-M3 Tabular Transformer
            logging.info("Training HIR-M3 Tabular Transformer...")
            try:
                config_hir = neural_hp.get('HIR-M3 Transformer', {'BATCH_SIZE': 64, 'EMBED_DIM': 32, 'NUM_HEADS': 4, 'HIDDEN_DIM': 128, 'LR': 1e-3, 'EPOCHS': 5, 'LAMBDA_HIR': 0.05, 'GAMMA': 0.5, 'PATIENCE': 3})
                prob = train_hir(X_train_tf.values, y_train, w_train, X_test_tf.values, y_test, w_test, tf_feature_cols, config_hir)
                if prob is not None: y_prob_dict['HIR-M3 Transformer'] = prob
            except Exception as e:
                logging.error(f"HIR-M3 Error: {e}")

            # Predict validation probabilities for threshold tuning (no test set leakage)
            X_tr, X_val, y_tr, y_val, w_tr_s, w_val_s = train_test_split(
                X_train, y_train, w_train, test_size=0.2, stratify=y_train, random_state=42
            )
            X_tr_tf, X_val_tf = X_tr[tf_feature_cols], X_val[tf_feature_cols]

            y_val_prob_dict = {}
            if 'Standard MLP' in y_prob_dict:
                try: y_val_prob_dict['Standard MLP'] = train_mlp(X_tr.values, y_tr, w_tr_s, X_val.values, y_val, w_val_s, config_mlp)
                except Exception: pass
            if 'Standard Transformer' in y_prob_dict:
                try: y_val_prob_dict['Standard Transformer'] = train_standard_transformer(X_tr_tf.values, y_tr, w_tr_s, X_val_tf.values, y_val, w_val_s, tf_feature_cols, config_trans)
                except Exception: pass
            if 'SAINT Transformer' in y_prob_dict:
                try: y_val_prob_dict['SAINT Transformer'] = train_saint(X_tr_tf.values, y_tr, w_tr_s, X_val_tf.values, y_val, w_val_s, tf_feature_cols, config_saint)
                except Exception: pass
            if 'HIR-M3 Transformer' in y_prob_dict:
                try: y_val_prob_dict['HIR-M3 Transformer'] = train_hir(X_tr_tf.values, y_tr, w_tr_s, X_val_tf.values, y_val, w_val_s, tf_feature_cols, config_hir)
                except Exception: pass

            # Evaluate neural models for this sub-cohort
            model_results_cohort = []
            boot_results_cohort = []

            for model_name, y_prob in y_prob_dict.items():
                if model_name in y_val_prob_dict and y_val_prob_dict[model_name] is not None:
                    thresh, _ = find_optimal_threshold(y_val, y_val_prob_dict[model_name], metric='f1')
                else:
                    thresh = 0.5

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

            del X_train, X_test, y_train, y_test
            gc.collect()

    # Save overall combined output CSV files
    if all_model_results:
        res_df = pd.DataFrame(all_model_results)
        res_path = os.path.join(output_dir, "urban_rural_neural_modeling_results.csv")
        res_df.to_csv(res_path, index=False)
        logging.info(f"\nSaved overall urban/rural neural modeling results to {res_path}")

    if all_bootstrap_results:
        boot_df = pd.DataFrame(all_bootstrap_results)
        boot_path = os.path.join(output_dir, "urban_rural_neural_bootstrap_results.csv")
        boot_df.to_csv(boot_path, index=False)
        logging.info(f"Saved overall urban/rural neural bootstrap results to {boot_path}")

    logging.info("=== PyTorch Neural Urban/Rural Pipeline Finished Successfully ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Neural Urban/Rural Pipeline (All 4 Models)")
    parser.add_argument("--cohort", default="all", choices=["all", "Texas", "Nationwide"], help="Target cohort execution")
    args = parser.parse_args()

    run_urban_rural_neural_pipeline(selected_cohort=args.cohort)
