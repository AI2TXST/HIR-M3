import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_CBWR"] = "COMPATIBLE"

import sys
import time
import gc
import json
import logging
import argparse
import traceback
import concurrent.futures
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add modeling directory to path
MODELING_DIR = os.path.dirname(os.path.abspath(__file__))
if MODELING_DIR not in sys.path:
    sys.path.insert(0, MODELING_DIR)

from models import (
    train_lgb, train_xgb, train_cb, train_rf, train_gbdt, train_logreg,
    train_mlp, train_standard_transformer, train_saint, train_hir,
    LGBM_AVAILABLE, XGB_AVAILABLE, CATBOOST_AVAILABLE, TORCH_AVAILABLE,
    LGBM_IMPORT_ERROR, XGB_IMPORT_ERROR, CATBOOST_IMPORT_ERROR, TORCH_IMPORT_ERROR
)
from metrics import calculate_metrics, calculate_bootstrap_ci, find_optimal_threshold

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensure safe PyTorch CPU thread management for multi-threaded worker pools
if TORCH_AVAILABLE:
    import torch
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

TARGET_COL = "ever_readmitted"

DATASETS = {
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

CONDITION_SUBGROUPS = {
    "Diabetes (d)": lambda d, hf, hyp: (d == 1),
    "Heart Failure (hf)": lambda d, hf, hyp: (hf == 1),
    "Hypertension (hyp)": lambda d, hf, hyp: (hyp == 1),
    "Diabetes + Heart Failure (d+hf)": lambda d, hf, hyp: (d == 1) & (hf == 1),
    "Diabetes + Hypertension (d+hyp)": lambda d, hf, hyp: (d == 1) & (hyp == 1),
    "Heart Failure + Hypertension (hf+hyp)": lambda d, hf, hyp: (hf == 1) & (hyp == 1),
    "Diabetes + Heart Failure + Hypertension (d+hf+hyp)": lambda d, hf, hyp: (d == 1) & (hf == 1) & (hyp == 1)
}

def resolve_condition_column(df, candidate_names):
    """Finds exact or case-insensitive column match."""
    for cand in candidate_names:
        if cand in df.columns:
            return cand
        matches = [c for c in df.columns if cand.lower() in c.lower()]
        if matches:
            return matches[0]
    return None

def load_and_prep_subgroup_data(filepath, subgroup_name, subgroup_filter_func, target_col=TARGET_COL):
    """
    Loads dataset, filters rows matching the specified condition subgroup combination,
    and returns scaled train/test matrices with zero ID leakage.
    """
    actual_path = next((p for p in filepath if os.path.exists(p)), None) if isinstance(filepath, list) else filepath
    if not actual_path or not os.path.exists(str(actual_path)):
        logging.error(f"Dataset path not found: {filepath}")
        return None

    if str(actual_path).endswith('.parquet'):
        logging.info(f"Loading dataset from Parquet: {actual_path}...")
        df = pd.read_parquet(actual_path)
    else:
        logging.info(f"Loading dataset in memory-efficient chunks from CSV: {actual_path}...")
        chunks = [chunk for chunk in pd.read_csv(actual_path, chunksize=100000, low_memory=False)]
        df = pd.concat(chunks, ignore_index=True)

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

    # Resolve condition columns
    diab_col = resolve_condition_column(df, ['has_diabetes', 'diab', 'Endocrine_Diabetes'])
    hf_col = resolve_condition_column(df, ['has_heart_failure', 'chf', 'heartfailure', 'Circulatory_HeartFailure'])
    hyp_col = resolve_condition_column(df, ['has_hypertension', 'hypertension', 'hyp', 'Circulatory_Hypertension'])

    if not (diab_col and hf_col and hyp_col):
        logging.error(f"Could not resolve all 3 condition columns in {actual_path}. Found: diab={diab_col}, hf={hf_col}, hyp={hyp_col}")
        return None

    # Apply Subgroup Filter
    mask = subgroup_filter_func(df[diab_col], df[hf_col], df[hyp_col])
    sub_df = df[mask].copy()

    initial_size = len(sub_df)
    if initial_size == 0:
        logging.error(f"Subgroup '{subgroup_name}' returned 0 matching rows.")
        return None

    logging.info(f"Filtered subgroup '{subgroup_name}': {initial_size:,} / {len(df):,} total rows.")

    # 50% Stratified Sample for Nationwide Cohort if > 50,000 rows
    if "TX" not in str(actual_path) and "texas" not in str(actual_path).lower():
        if len(sub_df) > 50000:
            logging.info(f"Sampling 50% of {len(sub_df):,} rows for Nationwide subgroup '{subgroup_name}'...")
            sub_df, _ = train_test_split(sub_df, train_size=0.50, stratify=sub_df[target_col], random_state=42)
            logging.info(f"Sampled Nationwide subgroup size: {len(sub_df):,} rows")

    total_sample_size = len(sub_df)

    # Drop non-feature columns, Beneficiary ID variants, and non-numeric object/string columns
    drop_cols = [
        target_col, 'ID', 'Patient_ID', 'ZipCode', 'Agency_Medicare_Number',
        'Facility_Internal_ID', 'COUNTY_NAME', 'BENE_ID', 'Beneficiary_ID',
        'Assessment_Effective_Date', 'COUNTYFIPS', 'Days_Cared_For', 'ever_deceased',
        'NumVisits', 'DaysBetweenVisits', 'PrevVisitDate', 'Last_Assessment_Date'
    ]
    id_cols = [c for c in sub_df.columns if 'beneficiary' in c.lower() or 'bene_id' in c.lower() or c.lower().endswith('id')]
    non_numeric = sub_df.select_dtypes(include=['object', 'string', 'category']).columns
    drop_cols = list(set(drop_cols).union(set(id_cols)).union(set(non_numeric)))

    feature_cols = [c for c in sub_df.columns if c not in drop_cols and not c.startswith('Unnamed')]

    # Coerce features to numeric float32
    X_df = sub_df[feature_cols].copy()
    for col in X_df.columns:
        if X_df[col].dtype == object or X_df[col].dtype == str or X_df[col].dtype.name == 'category':
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

    X_df = X_df.dropna(how='all', axis=1)
    X_df = X_df.fillna(X_df.median()).astype(np.float32)
    feature_cols = list(X_df.columns)

    X = X_df
    y = sub_df[target_col].values.astype(np.int8)

    # Train / Test Split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, total_sample_size

def load_tuned_hyperparameters(cohort_name, output_dir):
    """Loads saved tuned ML and Neural hyperparameter JSON files with robust path resolution."""
    raw_prefix = cohort_name.lower().replace(" ", "_")
    short_prefix = raw_prefix.replace("_cohort", "")
    prefixes = list(dict.fromkeys([raw_prefix, f"{short_prefix}_cohort", short_prefix]))

    search_dirs = [
        output_dir,
        os.path.join(MODELING_DIR, "results", "regular_modeling"),
        os.path.join(MODELING_DIR, "results"),
        os.path.join(MODELING_DIR, "..", "modeling", "results", "regular_modeling"),
        os.path.join("results", "regular_modeling"),
        os.path.join("modeling", "results", "regular_modeling"),
        os.path.join("..", "results", "regular_modeling")
    ]

    cand_ml, cand_neural = [], []
    for d in search_dirs:
        for p in prefixes:
            cand_ml.append(os.path.join(d, f"{p}_best_hyperparameters.json"))
            cand_neural.append(os.path.join(d, f"{p}_neural_best_hyperparameters.json"))

    ml_hp_path = next((p for p in cand_ml if os.path.exists(p)), None)
    neural_hp_path = next((p for p in cand_neural if os.path.exists(p)), None)

    ml_hp, neural_hp = {}, {}
    if ml_hp_path:
        try:
            with open(ml_hp_path) as f: ml_hp = json.load(f)
            logging.info(f"[HYPERPARAMS] Loaded ML hyperparameters for '{cohort_name}' from: {ml_hp_path}")
            logging.info(f"[HYPERPARAMS] Found tuned ML models: {list(ml_hp.keys())}")
        except Exception as e:
            logging.warning(f"[HYPERPARAMS WARNING] Failed loading ML hyperparameters {ml_hp_path}: {e}")
    else:
        logging.warning(f"[HYPERPARAMS WARNING] No ML hyperparameter JSON found for '{cohort_name}'. Using robust defaults.")

    if neural_hp_path:
        try:
            with open(neural_hp_path) as f: neural_hp = json.load(f)
            logging.info(f"[HYPERPARAMS] Loaded Neural hyperparameters for '{cohort_name}' from: {neural_hp_path}")
            logging.info(f"[HYPERPARAMS] Found tuned Neural models: {list(neural_hp.keys())}")
        except Exception as e:
            logging.warning(f"[HYPERPARAMS WARNING] Failed loading Neural hyperparameters {neural_hp_path}: {e}")
    else:
        logging.warning(f"[HYPERPARAMS WARNING] No Neural hyperparameter JSON found for '{cohort_name}'. Using robust defaults.")

    return ml_hp, neural_hp

def train_single_model_worker(model_name, X_tr, y_tr, w_tr, X_val, y_val, w_val, X_te, y_te, w_te, feature_cols, tf_feature_cols, ml_hp, neural_hp):
    """
    Worker function to train a single model, generate test prediction probabilities and validation probabilities.
    """
    start_t = time.time()
    prob_test, prob_val = None, None

    try:
        if model_name == 'Logistic Regression':
            cfg = ml_hp.get('Logistic Regression', {'C': 1.0})
            logging.info(f"  [Worker] Starting {model_name} (params: {cfg})")
            prob_test = train_logreg(X_tr.values, y_tr, w_tr, X_te.values, y_te, cfg)
            prob_val = train_logreg(X_tr.values, y_tr, w_tr, X_val.values, y_val, cfg)

        elif model_name == 'Random Forest':
            cfg = ml_hp.get('Random Forest', {'n_estimators': 100, 'max_depth': 15})
            logging.info(f"  [Worker] Starting {model_name} (params: {cfg})")
            prob_test = train_rf(X_tr.values, y_tr, w_tr, X_te.values, y_te, cfg)
            prob_val = train_rf(X_tr.values, y_tr, w_tr, X_val.values, y_val, cfg)

        elif model_name == 'Gradient Boosting':
            cfg = ml_hp.get('Gradient Boosting', {'n_estimators': 50, 'learning_rate': 0.05})
            logging.info(f"  [Worker] Starting {model_name} (params: {cfg})")
            prob_test = train_gbdt(X_tr.values, y_tr, w_tr, X_te.values, y_te, cfg)
            prob_val = train_gbdt(X_tr.values, y_tr, w_tr, X_val.values, y_val, cfg)

        elif model_name == 'LightGBM':
            if LGBM_AVAILABLE:
                cfg = ml_hp.get('LightGBM', {'learning_rate': 0.05, 'n_estimators': 100, 'num_leaves': 31})
                logging.info(f"  [Worker] Starting {model_name} (params: {cfg})")
                prob_test = train_lgb(X_tr.values, y_tr, w_tr, X_te.values, y_te, cfg)
                prob_val = train_lgb(X_tr.values, y_tr, w_tr, X_val.values, y_val, cfg)
            else:
                logging.warning(f"LightGBM not available: {LGBM_IMPORT_ERROR}")

        elif model_name == 'XGBoost':
            if XGB_AVAILABLE:
                cfg = ml_hp.get('XGBoost', {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100})
                logging.info(f"  [Worker] Starting {model_name} (params: {cfg})")
                prob_test = train_xgb(X_tr.values, y_tr, w_tr, X_te.values, y_te, cfg)
                prob_val = train_xgb(X_tr.values, y_tr, w_tr, X_val.values, y_val, cfg)
            else:
                logging.warning(f"XGBoost not available: {XGB_IMPORT_ERROR}")

        elif model_name == 'CatBoost':
            if CATBOOST_AVAILABLE:
                cfg = ml_hp.get('CatBoost', {'iterations': 100, 'learning_rate': 0.05, 'depth': 6})
                logging.info(f"  [Worker] Starting {model_name} (params: {cfg})")
                prob_test = train_cb(X_tr.values, y_tr, w_tr, X_te.values, y_te, cfg)
                prob_val = train_cb(X_tr.values, y_tr, w_tr, X_val.values, y_val, cfg)
            else:
                logging.warning(f"CatBoost not available: {CATBOOST_IMPORT_ERROR}")

        elif model_name == 'Standard MLP':
            if TORCH_AVAILABLE:
                cfg = dict(neural_hp.get('Standard MLP', {}))
                cfg.setdefault('BATCH_SIZE', 256)
                cfg.setdefault('HIDDEN_DIM', 128)
                cfg.setdefault('LR', 0.001)
                cfg.setdefault('EPOCHS', 5)
                cfg.setdefault('DROPOUT', 0.2)
                if cfg.get('BATCH_SIZE', 256) < 256: cfg['BATCH_SIZE'] = 256
                if cfg.get('EPOCHS', 5) > 5: cfg['EPOCHS'] = 5
                logging.info(f"  [Worker] Starting {model_name} (params: {cfg})")
                prob_test = train_mlp(X_tr.values, y_tr, w_tr, X_te.values, y_te, w_te, cfg)
                prob_val = train_mlp(X_tr.values, y_tr, w_tr, X_val.values, y_val, w_val, cfg)
            else:
                logging.warning(f"PyTorch not available for Standard MLP: {TORCH_IMPORT_ERROR}")

        elif model_name == 'Standard Transformer':
            if TORCH_AVAILABLE:
                cfg = dict(neural_hp.get('Standard Transformer', {}))
                cfg.setdefault('BATCH_SIZE', 256)
                cfg.setdefault('EMBED_DIM', 32)
                cfg.setdefault('NUM_HEADS', 4)
                cfg.setdefault('HIDDEN_DIM', 128)
                cfg.setdefault('LR', 1e-3)
                cfg.setdefault('EPOCHS', 3)
                if cfg.get('BATCH_SIZE', 256) < 256: cfg['BATCH_SIZE'] = 256
                if cfg.get('EPOCHS', 3) > 3: cfg['EPOCHS'] = 3
                logging.info(f"  [Worker] Starting {model_name} (params: {cfg})")
                X_tr_tf, X_val_tf, X_te_tf = X_tr[tf_feature_cols], X_val[tf_feature_cols], X_te[tf_feature_cols]
                prob_test = train_standard_transformer(X_tr_tf.values, y_tr, w_tr, X_te_tf.values, y_te, w_te, tf_feature_cols, cfg)
                prob_val = train_standard_transformer(X_tr_tf.values, y_tr, w_tr, X_val_tf.values, y_val, w_val, tf_feature_cols, cfg)
            else:
                logging.warning(f"PyTorch not available for Standard Transformer: {TORCH_IMPORT_ERROR}")

        elif model_name == 'SAINT Transformer':
            if TORCH_AVAILABLE:
                cfg = dict(neural_hp.get('SAINT Transformer', {}))
                cfg.setdefault('BATCH_SIZE', 256)
                cfg.setdefault('EMBED_DIM', 32)
                cfg.setdefault('NUM_HEADS', 4)
                cfg.setdefault('NUM_LAYERS', 1)
                cfg.setdefault('HIDDEN_DIM', 128)
                cfg.setdefault('LR', 1e-3)
                cfg.setdefault('EPOCHS', 3)
                cfg.setdefault('DROPOUT', 0.2)
                if cfg.get('BATCH_SIZE', 256) < 256: cfg['BATCH_SIZE'] = 256
                if cfg.get('EPOCHS', 3) > 3: cfg['EPOCHS'] = 3
                logging.info(f"  [Worker] Starting {model_name} (params: {cfg})")
                X_tr_tf, X_val_tf, X_te_tf = X_tr[tf_feature_cols], X_val[tf_feature_cols], X_te[tf_feature_cols]
                prob_test = train_saint(X_tr_tf.values, y_tr, w_tr, X_te_tf.values, y_te, w_te, tf_feature_cols, cfg)
                prob_val = train_saint(X_tr_tf.values, y_tr, w_tr, X_val_tf.values, y_val, w_val, tf_feature_cols, cfg)
            else:
                logging.warning(f"PyTorch not available for SAINT Transformer: {TORCH_IMPORT_ERROR}")

        elif model_name == 'HIR-M3 Transformer':
            if TORCH_AVAILABLE:
                cfg = dict(neural_hp.get('HIR-M3 Transformer', {}))
                cfg.setdefault('BATCH_SIZE', 256)
                cfg.setdefault('EMBED_DIM', 32)
                cfg.setdefault('NUM_HEADS', 4)
                cfg.setdefault('HIDDEN_DIM', 128)
                cfg.setdefault('LR', 1e-3)
                cfg.setdefault('EPOCHS', 3)
                cfg.setdefault('LAMBDA_HIR', 0.05)
                cfg.setdefault('GAMMA', 0.5)
                if cfg.get('BATCH_SIZE', 256) < 256: cfg['BATCH_SIZE'] = 256
                if cfg.get('EPOCHS', 3) > 3: cfg['EPOCHS'] = 3
                logging.info(f"  [Worker] Starting {model_name} (params: {cfg})")
                X_tr_tf, X_val_tf, X_te_tf = X_tr[tf_feature_cols], X_val[tf_feature_cols], X_te[tf_feature_cols]
                prob_test = train_hir(X_tr_tf.values, y_tr, w_tr, X_te_tf.values, y_te, w_te, tf_feature_cols, cfg)
                prob_val = train_hir(X_tr_tf.values, y_tr, w_tr, X_val_tf.values, y_val, w_val, tf_feature_cols, cfg)
            else:
                logging.warning(f"PyTorch not available for HIR-M3 Transformer: {TORCH_IMPORT_ERROR}")

    except Exception as e:
        logging.error(f"Error training {model_name}: {e}\n{traceback.format_exc()}")

    elapsed = time.time() - start_t
    logging.info(f"  [Worker] Finished {model_name} in {elapsed:.2f}s")

    if prob_test is not None:
        prob_test = np.asarray(prob_test).ravel()
    if prob_val is not None:
        prob_val = np.asarray(prob_val).ravel()

    return model_name, prob_test, prob_val

def run_condition_subgroup_pipeline(selected_cohort="all", max_workers=4):
    logging.info("=========================================================================")
    logging.info("  STARTING CONDITION SUBGROUP MODELING PIPELINE (ALL 10 MODELS)")
    logging.info("=========================================================================")
    logging.info(f"Parallel Workers: {max_workers}")
    logging.info(f"Evaluated Subgroups: {list(CONDITION_SUBGROUPS.keys())}")
    logging.info(f"Package Availability -> LightGBM: {LGBM_AVAILABLE}, XGBoost: {XGB_AVAILABLE}, CatBoost: {CATBOOST_AVAILABLE}, PyTorch: {TORCH_AVAILABLE}")

    if not TORCH_AVAILABLE:
        logging.error(f"CRITICAL WARNING: PyTorch is NOT available ({TORCH_IMPORT_ERROR}). Neural models will be skipped!")

    output_dir = os.path.join(MODELING_DIR, "results", "condition_subgroups")
    os.makedirs(output_dir, exist_ok=True)

    all_model_results = []
    all_bootstrap_results = []

    if selected_cohort.lower().startswith("tx") or selected_cohort.lower().startswith("texas"):
        target_files = {k: v for k, v in DATASETS.items() if "Texas" in k}
    elif selected_cohort.lower().startswith("nat") or selected_cohort.lower().startswith("nationwide"):
        target_files = {k: v for k, v in DATASETS.items() if "Nationwide" in k}
    else:
        target_files = DATASETS

    # Group models into Baseline ML (Parallel) and Neural Models (Sequential for MKL thread safety)
    ml_models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'LightGBM', 'XGBoost', 'CatBoost']
    neural_models = ['Standard MLP', 'Standard Transformer', 'SAINT Transformer', 'HIR-M3 Transformer'] if TORCH_AVAILABLE else []

    for cohort_name, filepath in target_files.items():
        ml_hp, neural_hp = load_tuned_hyperparameters(cohort_name, output_dir)

        for subgroup_name, subgroup_filter_func in CONDITION_SUBGROUPS.items():
            full_subgroup_name = f"{cohort_name} - {subgroup_name}"
            logging.info(f"\n=========================================================")
            logging.info(f"   PROCESSING SUB-COHORT: {full_subgroup_name}")
            logging.info(f"=========================================================")

            data = load_and_prep_subgroup_data(filepath, subgroup_name, subgroup_filter_func)
            if data is None:
                continue

            X_train, X_test, y_train, y_test, feature_cols, cohort_sample_size = data
            w_train = np.ones_like(y_train, dtype=float)
            w_test = np.ones_like(y_test, dtype=float)

            # Feature Subsetting for Tabular Transformers (OOM Memory Safety)
            if len(feature_cols) > 200:
                variances = X_train.var(axis=0).values
                top_200_idx = np.argsort(variances)[::-1][:200]
                tf_feature_cols = [feature_cols[i] for i in top_200_idx]
            else:
                tf_feature_cols = feature_cols

            # Predict validation probabilities for threshold tuning (no test set leakage)
            X_tr, X_val, y_tr, y_val, w_tr_s, w_val_s = train_test_split(
                X_train, y_train, w_train, test_size=0.2, stratify=y_train, random_state=42
            )

            y_prob_dict = {}
            y_val_prob_dict = {}

            # Step 1: Execute 6 Baseline ML models in parallel ThreadPool
            logging.info(f"Executing {len(ml_models)} Baseline ML models in parallel (workers={max_workers})...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_model = {
                    executor.submit(
                        train_single_model_worker,
                        m_name, X_tr, y_tr, w_tr_s, X_val, y_val, w_val_s, X_test, y_test, w_test,
                        feature_cols, tf_feature_cols, ml_hp, neural_hp
                    ): m_name for m_name in ml_models
                }

                for future in concurrent.futures.as_completed(future_to_model):
                    m_name = future_to_model[future]
                    try:
                        name, p_test, p_val = future.result()
                        if p_test is not None: y_prob_dict[name] = p_test
                        if p_val is not None: y_val_prob_dict[name] = p_val
                    except Exception as exc:
                        logging.error(f"ML Model '{m_name}' failed with exception: {exc}")

            # Step 2: Execute PyTorch Neural & Transformer models sequentially (prevents MKL Bernoulli RNG multi-thread collision)
            if TORCH_AVAILABLE:
                logging.info(f"Executing {len(neural_models)} PyTorch Neural & Transformer models sequentially for MKL thread safety...")
                for n_name in neural_models:
                    try:
                        name, p_test, p_val = train_single_model_worker(
                            n_name, X_tr, y_tr, w_tr_s, X_val, y_val, w_val_s, X_test, y_test, w_test,
                            feature_cols, tf_feature_cols, ml_hp, neural_hp
                        )
                        if p_test is not None: y_prob_dict[name] = p_test
                        if p_val is not None: y_val_prob_dict[name] = p_val
                    except Exception as exc:
                        logging.error(f"Neural Model '{n_name}' failed with exception: {exc}")

            # Evaluate all successfully trained models for this subgroup
            subgroup_model_results = []
            subgroup_boot_results = []

            for model_name, y_prob in y_prob_dict.items():
                if model_name in y_val_prob_dict and y_val_prob_dict[model_name] is not None:
                    thresh, _ = find_optimal_threshold(y_val, y_val_prob_dict[model_name], metric='f1')
                else:
                    thresh = 0.5

                m = calculate_metrics(y_test, y_prob, threshold=thresh)
                m['Cohort'] = cohort_name
                m['Subgroup'] = subgroup_name
                m['Cohort Size'] = cohort_sample_size
                m['Num Features'] = len(feature_cols)
                m['Model'] = model_name
                all_model_results.append(m)
                subgroup_model_results.append(m)

                # Compute Bootstrap 95% CIs
                boot_summary = calculate_bootstrap_ci(y_test, y_prob, threshold=thresh, n_bootstrap=200)
                for m_name, b_data in boot_summary.items():
                    b_item = {
                        'Cohort': cohort_name,
                        'Subgroup': subgroup_name,
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
                    subgroup_boot_results.append(b_item)

            # Save subgroup-specific output files
            clean_subgroup_name = subgroup_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_plus_")
            sub_prefix = f"{cohort_name.replace(' ', '_')}_{clean_subgroup_name}"
            if subgroup_model_results:
                pd.DataFrame(subgroup_model_results).to_csv(os.path.join(output_dir, f"{sub_prefix}_modeling_results.csv"), index=False)
            if subgroup_boot_results:
                pd.DataFrame(subgroup_boot_results).to_csv(os.path.join(output_dir, f"{sub_prefix}_bootstrap_results.csv"), index=False)

            del X_train, X_test, y_train, y_test
            gc.collect()

    # Save aggregated output across all cohorts & subgroups
    if all_model_results:
        res_df = pd.DataFrame(all_model_results)
        res_path = os.path.join(output_dir, "condition_subgroups_modeling_results.csv")
        res_df.to_csv(res_path, index=False)
        logging.info(f"\nSaved aggregated condition subgroup modeling results to {res_path}")

    if all_bootstrap_results:
        boot_df = pd.DataFrame(all_bootstrap_results)
        boot_path = os.path.join(output_dir, "condition_subgroups_bootstrap_results.csv")
        boot_df.to_csv(boot_path, index=False)
        logging.info(f"Saved aggregated condition subgroup bootstrap results to {boot_path}")

    logging.info("=========================================================================")
    logging.info("  CONDITION SUBGROUP MODELING PIPELINE COMPLETED SUCCESSFULLY")
    logging.info("=========================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Condition Subgroup Modeling Pipeline")
    parser.add_argument("--cohort", choices=["Texas", "Nationwide", "all"], default="all",
                        help="Specify cohort to train (Texas, Nationwide, or all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker threads for ML models (default: 4)")
    args = parser.parse_args()
    run_condition_subgroup_pipeline(args.cohort, max_workers=args.workers)
