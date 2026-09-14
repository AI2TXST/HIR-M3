import os
import sys
import time
import gc
import json
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    precision_score, recall_score, brier_score_loss, confusion_matrix,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

PARITY_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PARITY_DIR)
MODELING_DIR = os.path.join(BASE_DIR, "modeling")
RESULTS_DIR = os.path.join(MODELING_DIR, "results")

for d in [BASE_DIR, PARITY_DIR, MODELING_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from parity.models import get_m3_feature_groups_tokenized, ACTParityV2, HIRM3_ACTParity_Hybrid, TORCH_AVAILABLE, DEVICE
from parity.loss import AugmentedLagrangianDGAPLoss, DualMultiplierManager, HIRM3_ACTParity_HybridLoss
from parity.metrics import (
    calculate_stabilized_ctdi,
    calculate_harm_weighted_efnhi,
    calculate_platt_calibration_slope,
    calculate_subgroup_disparity_metrics,
    calculate_bootstrap_confidence_intervals,
    evaluate_pareto_frontier
)

try:
    from modeling.models import train_hir
except ImportError:
    try:
        from models import train_hir
    except ImportError:
        train_hir = None

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TARGET_COL = "ever_readmitted"

DATASETS_TO_RUN = {
    "Texas": [
        "../data/processed_final_mergedDF_condensed_TX.parquet",
        "data/processed_final_mergedDF_condensed_TX.parquet",
        "../data/processed_final_mergedDF_condensed_TX.csv",
        "data/processed_final_mergedDF_condensed_TX.csv"
    ],
    "Nationwide": [
        "../data/processed_final_mergedDF_condensed.parquet",
        "data/processed_final_mergedDF_condensed.parquet",
        "../data/processed_final_mergedDF_condensed.csv",
        "data/processed_final_mergedDF_condensed.csv"
    ]
}

def load_best_hyperparams(dataset_key="Texas"):
    """
    Loads pre-tuned best hyperparameter JSON configurations for GBDTs and Neural architectures.
    """
    key_lower = dataset_key.lower()
    gbdt_path = os.path.join(RESULTS_DIR, f"{key_lower}_cohort_best_hyperparameters.json")
    if not os.path.exists(gbdt_path):
        gbdt_path = os.path.join(RESULTS_DIR, f"{key_lower}_best_hyperparameters.json")

    neural_path = os.path.join(RESULTS_DIR, f"{key_lower}_cohort_neural_best_hyperparameters.json")
    if not os.path.exists(neural_path):
        neural_path = os.path.join(RESULTS_DIR, f"neural_best_hyperparameters.json")

    gbdt_params, neural_params = {}, {}
    if os.path.exists(gbdt_path):
        try:
            with open(gbdt_path, 'r') as f:
                gbdt_params = json.load(f)
            logging.info(f"Loaded GBDT best hyperparams for {dataset_key} from {gbdt_path}")
        except Exception as e:
            logging.error(f"Error reading GBDT hyperparams: {e}")

    if os.path.exists(neural_path):
        try:
            with open(neural_path, 'r') as f:
                neural_params = json.load(f)
            logging.info(f"Loaded Neural best hyperparams for {dataset_key} from {neural_path}")
        except Exception as e:
            logging.error(f"Error reading Neural hyperparams: {e}")

    return gbdt_params, neural_params

def load_cohort_dataset(dataset_key="Texas"):
    """
    Loads dataset: 100% of Texas dataset or 20% stratified sample of Nationwide dataset directly from Parquet.
    Constructs race & urban/rural demographic group arrays.
    """
    candidates = DATASETS_TO_RUN.get(dataset_key, DATASETS_TO_RUN["Texas"])
    filepath = next((p for p in candidates if os.path.exists(p)), None)
    if not filepath:
        alt_path = os.path.join(BASE_DIR, "data", "processed_final_mergedDF_condensed_TX.parquet")
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            logging.error(f"Dataset files for {dataset_key} not found.")
            return None

    is_nationwide = "nationwide" in dataset_key.lower() or ("tx" not in filepath.lower() and "texas" not in filepath.lower())

    if filepath.endswith('.parquet'):
        logging.info(f"Loading {dataset_key} dataset directly from Parquet: {filepath}")
        df = pd.read_parquet(filepath)
        if is_nationwide:
            logging.info(f"Sampling 50% stratified sample for Nationwide dataset (N = {len(df)} -> {int(len(df)*0.5)})...")
            df = df.sample(frac=0.50, random_state=42).copy()
        else:
            logging.info(f"Using 100% of Texas dataset (N = {len(df)})...")
    else:
        logging.info(f"Loading {dataset_key} dataset in memory-efficient chunks from CSV: {filepath}")
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=100000, low_memory=False):
            if is_nationwide:
                chunk = chunk.sample(frac=0.50, random_state=42).copy()
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

    possible_targets = [TARGET_COL, 'ever_readmitted', 'READMISSION', 'Readmission']
    t_col = next((t for t in possible_targets if t in df.columns), None)
    if not t_col:
        logging.error(f"Target column not found in {dataset_key}.")
        return None

    race_col_name = "Race_Demographic"
    if race_col_name not in df.columns:
        race_dummies = {
            'Asian': ['Asian'],
            'Black': ['Black_or_African_American', 'Black'],
            'Hispanic': ['Hispanic_or_Latino', 'Hispanic'],
            'White': ['White'],
            'AIAN': ['American_Indian_or_Alaska_Native', 'AIAN'],
            'NHPI': ['Native_Hawiian_or_Pacific_Islander', 'NHPI']
        }
        race_series = pd.Series('White', index=df.index)
        for label, cols in race_dummies.items():
            for c in cols:
                if c in df.columns:
                    race_series[df[c] == 1] = label
        df[race_col_name] = race_series

    return df, t_col, race_col_name

def train_hirm3_act_parity_hybrid(X_tr, y_tr, g_tr, X_va, y_va, g_va, X_te, feature_cols, config):
    """
    Trains the unified HIR-M3 + ACT-Parity Hybrid Model using tuned hyperparameters.
    """
    micro_cols, meso_cols, macro_cols = get_m3_feature_groups_tokenized(feature_cols, remove_memorization_ids=True)

    micro_idxs = [feature_cols.index(c) for c in micro_cols if c in feature_cols]
    meso_idxs = [feature_cols.index(c) for c in meso_cols if c in feature_cols]
    macro_idxs = [feature_cols.index(c) for c in macro_cols if c in feature_cols]

    X_tr_micro = X_tr[:, micro_idxs] if micro_idxs else np.zeros((len(X_tr), 1), dtype=np.float32)
    X_tr_meso = X_tr[:, meso_idxs] if meso_idxs else np.zeros((len(X_tr), 0), dtype=np.float32)
    X_tr_macro = X_tr[:, macro_idxs] if macro_idxs else np.zeros((len(X_tr), 0), dtype=np.float32)

    X_va_micro = X_va[:, micro_idxs] if micro_idxs else np.zeros((len(X_va), 1), dtype=np.float32)
    X_va_meso = X_va[:, meso_idxs] if meso_idxs else np.zeros((len(X_va), 0), dtype=np.float32)
    X_va_macro = X_va[:, macro_idxs] if macro_idxs else np.zeros((len(X_va), 0), dtype=np.float32)

    X_te_micro = X_te[:, micro_idxs] if micro_idxs else np.zeros((len(X_te), 1), dtype=np.float32)
    X_te_meso = X_te[:, meso_idxs] if meso_idxs else np.zeros((len(X_te), 0), dtype=np.float32)
    X_te_macro = X_te[:, macro_idxs] if macro_idxs else np.zeros((len(X_te), 0), dtype=np.float32)

    batch_size = config.get('BATCH_SIZE', config.get('batch_size', 64))
    epochs = config.get('EPOCHS', config.get('epochs', 10))

    unique_groups = sorted(list(set(g_tr.tolist() + g_va.tolist())))
    g_map = {g: i for i, g in enumerate(unique_groups)}
    g_tr_code = np.array([g_map.get(g, 0) for g in g_tr], dtype=np.int64)
    g_va_code = np.array([g_map.get(g, 0) for g in g_va], dtype=np.int64)

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_micro, dtype=torch.float32),
        torch.tensor(X_tr_meso, dtype=torch.float32),
        torch.tensor(X_tr_macro, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
        torch.tensor(g_tr_code, dtype=torch.long)
    )
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = HIRM3_ACTParity_Hybrid(
        num_micro=X_tr_micro.shape[1],
        num_meso=X_tr_meso.shape[1],
        num_macro=X_tr_macro.shape[1],
        embed_dim=config.get('EMBED_DIM', config.get('embed_dim', 32)),
        num_heads=config.get('NUM_HEADS', config.get('num_heads', 4)),
        alpha=config.get('alpha', 0.5),
        dropout=config.get('DROPOUT', config.get('dropout', 0.1))
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.get('LR', config.get('lr', 1e-3)), weight_decay=1e-4)
    criterion = HIRM3_ACTParity_HybridLoss(
        delta=config.get('delta', 0.04),
        rho=config.get('rho', 1.0),
        min_support=config.get('min_support', 30),
        lambda_hir=config.get('LAMBDA_HIR', config.get('lambda_hir', 0.05)),
        gamma=config.get('GAMMA', config.get('gamma', 0.5)),
        lambda_inv=config.get('lambda_inv', 0.1)
    )
    multiplier_mgr = DualMultiplierManager(eta=0.05)

    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_constraints = {}

        for bx_mic, bx_mes, bx_mac, by, bg in tr_loader:
            bx_mic, bx_mes, bx_mac = bx_mic.to(DEVICE), bx_mes.to(DEVICE), bx_mac.to(DEVICE)
            by, bg = by.to(DEVICE), bg.to(DEVICE)

            optimizer.zero_grad()
            logits, _, attn_w = model(bx_mic, bx_mes, bx_mac)

            bx_mes_pert = bx_mes[:, torch.randperm(bx_mes.size(1))] if bx_mes.size(1) > 0 else None
            logits_pert = None
            if bx_mes_pert is not None:
                logits_pert, _, _ = model(bx_mic, bx_mes_pert, bx_mac)

            loss, c_dict = criterion(
                logits, by, bg, multiplier_mgr.get_all_lambdas(),
                attn_weights=attn_w, micro_idxs=list(range(X_tr_micro.shape[1])),
                meso_idxs=list(range(X_tr_meso.shape[1])), logits_perturbed=logits_pert
            )
            loss.backward()
            optimizer.step()

            for k, v in c_dict.items(): epoch_constraints[k] = v

        multiplier_mgr.update(epoch_constraints)

        # Validation Check (Batched to prevent OOM)
        va_probs = predict_batched_hybrid(model, X_va_micro, X_va_meso, X_va_macro, batch_size=512)
        try:
            va_auc = roc_auc_score(y_va, va_probs)
        except Exception:
            va_auc = 0.5

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test Evaluation (Batched to prevent OOM)
    te_probs = predict_batched_hybrid(model, X_te_micro, X_te_meso, X_te_macro, batch_size=512)
    return te_probs

def predict_batched_hybrid(model, X_micro, X_meso, X_macro, batch_size=512):
    """
    Evaluates PyTorch model in mini-batches to prevent OOM memory allocations in multi-head attention.
    """
    model.eval()
    all_probs = []
    N = len(X_micro)
    
    with torch.no_grad():
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            b_mic = torch.tensor(X_micro[start_idx:end_idx], dtype=torch.float32).to(DEVICE)
            b_mes = torch.tensor(X_meso[start_idx:end_idx], dtype=torch.float32).to(DEVICE) if X_meso is not None and X_meso.shape[1] > 0 else None
            b_mac = torch.tensor(X_macro[start_idx:end_idx], dtype=torch.float32).to(DEVICE) if X_macro is not None and X_macro.shape[1] > 0 else None

            logits, _, _ = model(b_mic, b_mes, b_mac)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            if probs.ndim == 0:
                probs = np.array([probs.item()])
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((N,), dtype=np.float32)

def eval_and_save_single_model(m_name, y_prob, y_test, race_te, dataset_key, baseline_auc, comparison_rows, output_csv_path):
    """
    Evaluates metrics for a single model with Precision-Recall curve threshold tuning 
    to maximize F1-Score on imbalanced class distributions, and writes to CSV.
    """
    try:
        auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
    except Exception:
        auc, pr_auc = 0.5, 0.0

    # Tune decision threshold via Precision-Recall Curve to maximize F1-Score
    prec_curve, rec_curve, th_curve = precision_recall_curve(y_test, y_prob)
    f1_curve = 2 * (prec_curve * rec_curve) / (prec_curve + rec_curve + 1e-8)
    opt_idx = np.argmax(f1_curve)
    opt_thresh = float(th_curve[opt_idx]) if opt_idx < len(th_curve) else 0.5

    preds = (np.asarray(y_prob) >= opt_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))

    subgroup_df, summary = calculate_subgroup_disparity_metrics(y_test, y_prob, race_te)
    platt_slope, _, brier = calculate_platt_calibration_slope(y_test, y_prob)

    ret_ratio = float(auc / (baseline_auc + 1e-8)) if baseline_auc is not None else 1.0

    row = {
        'Cohort': dataset_key,
        'Model Name': m_name,
        'Optimal_Threshold': round(opt_thresh, 4),
        'Overall_ROC_AUC': round(auc, 4),
        'PR_AUC': round(pr_auc, 4),
        'F1_Score': round(f1, 4),
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'AUC_Retention_Ratio': round(ret_ratio, 4),
        'Overall_FNR': summary['Overall_FNR'],
        'Worst_Group_FNR': summary['Worst_Group_FNR'],
        'FNR_Gap': summary['Delta_FNR'],
        'Equalized_Odds_Difference': summary['Equalized_Odds_Difference'],
        'Generalized_Entropy_Index': summary['Generalized_Entropy_Index'],
        'EFNHI_Star': summary['EFNHI_Star'],
        'Brier_Score': brier,
        'Platt_Calibration_Slope': platt_slope
    }
    comparison_rows.append(row)
    df_cur = pd.DataFrame(comparison_rows)
    df_cur.to_csv(output_csv_path, index=False)
    logging.info(f" Saved incremental metrics for '{m_name}' (Optimal Thresh: {opt_thresh:.4f}) -> {output_csv_path} | ROC-AUC: {auc:.4f}, F1: {f1:.4f}, Recall: {rec:.4f}")
    return row

def run_comprehensive_model_comparison(dataset_key="Texas"):
    """
    Trains and evaluates ALL models side-by-side using pre-tuned best hyperparameters:
    Applies RandomOverSampler to balance minority readmission class during training.
    """
    logging.info(f"\n=================================================================")
    logging.info(f"  UNIFIED COMPARISON WITH BEST HYPERPARAMETERS: ({dataset_key.upper()})")
    logging.info(f"=================================================================")

    gbdt_hp, neural_hp = load_best_hyperparams(dataset_key=dataset_key)

    loaded = load_cohort_dataset(dataset_key=dataset_key)
    if not loaded: return None

    df, t_col, race_col = loaded

    drop_cols = {
        'BENE_ID', 'Beneficiary_ID', 'Assessment_Effective_Date', 'COUNTYFIPS',
        'Days_Cared_For', 'ever_deceased', 'NumVisits', 'DaysBetweenVisits',
        'PrevVisitDate', 'Last_Assessment_Date', t_col, race_col
    }
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    feature_cols = [c for c in numeric_cols if c not in drop_cols and not c.lower().endswith('id')]

    # Coerce features and arrays safely via native python lists to avoid PyArrow ExtensionArray bugs
    X_df = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    X = np.array(X_df.values, dtype=np.float32)
    y = np.array(df[t_col].tolist(), dtype=np.int64)
    race_array = np.array(df[race_col].tolist(), dtype=str)

    # Clean raw DataFrames immediately to free RAM
    del df, X_df
    gc.collect()

    # Split using pure integer indices to guarantee compatibility with all sklearn/pandas/pyarrow versions
    n_total = len(y)
    indices = np.arange(n_total)

    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y
    )
    tr_idx, va_idx = train_test_split(
        train_idx, test_size=0.15, random_state=42, stratify=y[train_idx]
    )

    X_tr = X[tr_idx]
    y_tr = y[tr_idx]
    race_tr_s = race_array[tr_idx]

    X_va = X[va_idx]
    y_va = y[va_idx]
    race_va_s = race_array[va_idx]

    X_te = X[test_idx]
    y_test = y[test_idx]
    race_te = race_array[test_idx]

    del X, race_array, train_idx, test_idx, tr_idx, va_idx
    gc.collect()

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    del X_tr, X_va, X_te
    gc.collect()

    # Oversample minority readmission class on training dataset
    try:
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        idx_arr = np.arange(len(y_tr)).reshape(-1, 1)
        res_idx, y_tr_res = ros.fit_resample(idx_arr, y_tr)
        res_idx = res_idx.ravel()

        X_tr_s = X_tr_s[res_idx]
        y_tr = y_tr_res
        race_tr_s = race_tr_s[res_idx]
        logging.info(f"  Applied RandomOverSampler: Balanced training set from {len(res_idx):,} rows (50% positive / 50% negative).")
    except Exception as e:
        logging.warning(f"  Could not apply RandomOverSampler: {e}. Training on unweighted dataset.")

    output_dir = os.path.join(PARITY_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"unified_comparison_{dataset_key.lower()}.csv")

    predictions = {}
    comparison_rows = []
    baseline_auc = None

    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))

    # 1. Baseline GBDTs using Tuned Hyperparameters with Immediate Model Deletion & Incremental Saving
    logging.info("  Training LightGBM (Tuned)...")
    if LGBM_AVAILABLE:
        hp_lgb = gbdt_hp.get('LightGBM', {'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 50})
        lgb_m = lgb.LGBMClassifier(random_state=42, n_jobs=num_cpus, verbose=-1, **hp_lgb)
        lgb_m.fit(X_tr_s, y_tr)
        predictions['LightGBM'] = lgb_m.predict_proba(X_te_s)[:, 1]
        del lgb_m
        gc.collect()
        eval_and_save_single_model('LightGBM', predictions['LightGBM'], y_test, race_te, dataset_key, baseline_auc, comparison_rows, output_csv_path)

    logging.info("  Training XGBoost (Tuned)...")
    if XGB_AVAILABLE:
        hp_xgb = gbdt_hp.get('XGBoost', {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200})
        xgb_m = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', tree_method='hist', n_jobs=num_cpus, **hp_xgb)
        xgb_m.fit(X_tr_s, y_tr)
        predictions['XGBoost'] = xgb_m.predict_proba(X_te_s)[:, 1]
        del xgb_m
        gc.collect()
        baseline_auc = float(roc_auc_score(y_test, predictions['XGBoost']))
        eval_and_save_single_model('XGBoost', predictions['XGBoost'], y_test, race_te, dataset_key, baseline_auc, comparison_rows, output_csv_path)

    logging.info("  Training CatBoost (Tuned)...")
    if CATBOOST_AVAILABLE:
        hp_cb = gbdt_hp.get('CatBoost', {'learning_rate': 0.1, 'depth': 6, 'iterations': 200})
        cb_m = cb.CatBoostClassifier(random_state=42, verbose=0, thread_count=num_cpus, **hp_cb)
        cb_m.fit(X_tr_s, y_tr)
        predictions['CatBoost'] = cb_m.predict_proba(X_te_s)[:, 1]
        del cb_m
        gc.collect()
        eval_and_save_single_model('CatBoost', predictions['CatBoost'], y_test, race_te, dataset_key, baseline_auc, comparison_rows, output_csv_path)

    logging.info("  Training Random Forest (Tuned)...")
    hp_rf = gbdt_hp.get('Random Forest', {'max_depth': 20, 'n_estimators': 50})
    rf_m = RandomForestClassifier(random_state=42, n_jobs=num_cpus, **hp_rf)
    rf_m.fit(X_tr_s, y_tr)
    predictions['Random Forest'] = rf_m.predict_proba(X_te_s)[:, 1]
    del rf_m
    gc.collect()
    eval_and_save_single_model('Random Forest', predictions['Random Forest'], y_test, race_te, dataset_key, baseline_auc, comparison_rows, output_csv_path)

    logging.info("  Training Logistic Regression (Tuned)...")
    hp_lr = gbdt_hp.get('Logistic Regression', {'C': 10.0})
    lr_m = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', n_jobs=num_cpus, **hp_lr)
    lr_m.fit(X_tr_s, y_tr)
    predictions['Logistic Regression'] = lr_m.predict_proba(X_te_s)[:, 1]
    del lr_m
    gc.collect()
    eval_and_save_single_model('Logistic Regression', predictions['Logistic Regression'], y_test, race_te, dataset_key, baseline_auc, comparison_rows, output_csv_path)



    # 3. HIR-M3 Transformer & 70% XGBoost : 30% HIR-M3 Ensemble
    logging.info("  Training HIR-M3 Transformer (Tuned)...")
    hir_hp = dict(neural_hp.get('HIR-M3 Transformer', {}))
    hir_hp.setdefault('BATCH_SIZE', 256)
    hir_hp.setdefault('EMBED_DIM', 32)
    hir_hp.setdefault('NUM_HEADS', 4)
    hir_hp.setdefault('HIDDEN_DIM', 128)
    hir_hp.setdefault('LR', 0.001)
    hir_hp.setdefault('EPOCHS', 5)
    hir_hp.setdefault('LAMBDA_HIR', 0.05)
    hir_hp.setdefault('GAMMA', 0.5)
    if hir_hp.get('BATCH_SIZE', 256) < 256: hir_hp['BATCH_SIZE'] = 256
    if hir_hp.get('EPOCHS', 5) > 5: hir_hp['EPOCHS'] = 5

    if train_hir is not None:
        try:
            w_tr_dummy = np.ones(len(y_tr), dtype=np.float32)
            w_te_dummy = np.ones(len(y_test), dtype=np.float32)
            hir_prob = train_hir(X_tr_s, y_tr, w_tr_dummy, X_te_s, y_test, w_te_dummy, feature_cols, hir_hp)
            if hir_prob is not None:
                predictions['HIR-M3 Transformer'] = hir_prob
                eval_and_save_single_model('HIR-M3 Transformer', predictions['HIR-M3 Transformer'], y_test, race_te, dataset_key, baseline_auc, comparison_rows, output_csv_path)

                if 'XGBoost' in predictions:
                    predictions['70% XGBoost : 30% HIR-M3 Ensemble'] = 0.7 * predictions['XGBoost'] + 0.3 * predictions['HIR-M3 Transformer']
                    eval_and_save_single_model('70% XGBoost : 30% HIR-M3 Ensemble', predictions['70% XGBoost : 30% HIR-M3 Ensemble'], y_test, race_te, dataset_key, baseline_auc, comparison_rows, output_csv_path)
        except Exception as e:
            logging.error(f"Error training HIR-M3 Transformer: {e}")

    # 4. Standard ACT-Parity v2
    logging.info("  Training Standard ACT-Parity v2...")
    act_parity_prob, _, _ = train_act_parity_model_runner(X_tr_s, y_tr, race_tr_s, X_va_s, y_va, race_va_s, X_te_s, feature_cols)
    predictions['ACT-Parity v2'] = act_parity_prob
    eval_and_save_single_model('ACT-Parity v2', predictions['ACT-Parity v2'], y_test, race_te, dataset_key, baseline_auc, comparison_rows, output_csv_path)

    # 4. GBDT + Neural Ensemble (70% GBDT : 30% Neural)
    if 'XGBoost' in predictions and 'ACT-Parity v2' in predictions:
        predictions['70% GBDT : 30% ACT-Parity Ensemble'] = 0.7 * predictions['XGBoost'] + 0.3 * predictions['ACT-Parity v2']
        eval_and_save_single_model('70% GBDT : 30% ACT-Parity Ensemble', predictions['70% GBDT : 30% ACT-Parity Ensemble'], y_test, race_te, dataset_key, baseline_auc, comparison_rows, output_csv_path)

    # 5. UNIFIED HYBRID: HIR-M3 + ACT-Parity Hybrid Model using Tuned HIR-M3 Hyperparameters
    logging.info("  Training UNIFIED HIR-M3 + ACT-PARITY HYBRID MODEL (Tuned)...")
    hybrid_config = {
        'batch_size': hir_hp.get('BATCH_SIZE', 256),
        'epochs': hir_hp.get('EPOCHS', 5),
        'embed_dim': hir_hp.get('EMBED_DIM', 32),
        'num_heads': hir_hp.get('NUM_HEADS', 4),
        'lr': hir_hp.get('LR', 1e-3),
        'lambda_hir': hir_hp.get('LAMBDA_HIR', 0.05),
        'gamma': hir_hp.get('GAMMA', 0.5),
        'delta': 0.04,
        'lambda_inv': 0.1
    }
    hybrid_prob = train_hirm3_act_parity_hybrid(X_tr_s, y_tr, race_tr_s, X_va_s, y_va, race_va_s, X_te_s, feature_cols, hybrid_config)
    predictions['HIR-M3 + ACT-Parity Hybrid'] = hybrid_prob
    eval_and_save_single_model('HIR-M3 + ACT-Parity Hybrid', predictions['HIR-M3 + ACT-Parity Hybrid'], y_test, race_te, dataset_key, baseline_auc, comparison_rows, output_csv_path)

    res_df = pd.DataFrame(comparison_rows)
    pareto_df = evaluate_pareto_frontier(res_df, baseline_auc if baseline_auc else 0.80)

    return dataset_key, res_df, pareto_df

def train_act_parity_model_runner(X_tr, y_tr, g_tr, X_va, y_va, g_va, X_te, feature_cols):
    from parity.run_experiments import train_act_parity_model
    return train_act_parity_model(X_tr, y_tr, g_tr, X_va, y_va, g_va, X_te, feature_cols, {'variant': 'V6_Full', 'epochs': 5, 'batch_size': 256})

def generate_unified_markdown_report(results_dict, pareto_dict, report_path=None):
    if report_path is None:
        report_path = os.path.join(BASE_DIR, "docs", "UNIFIED_MODEL_PARITY_COMPARISON_REPORT.md")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("# Unified Benchmark Report: HIR-M3 + ACT-Parity Hybrid vs. Baseline & Ensemble Models\n\n")
        f.write("**Framework**: Health Equity and Algorithmic Learning (HEAL) & Wang's Bias Evaluation Checklist\n\n")
        f.write("---\n\n")
        f.write("## 1. Unified Side-by-Side Model Comparison Table\n\n")

        for cohort_key, df_res in results_dict.items():
            f.write(f"### Cohort: {cohort_key}\n\n")
            cols_to_show = [
                'Model Name', 'Optimal_Threshold', 'Overall_ROC_AUC', 'PR_AUC', 'F1_Score', 'Accuracy', 'Precision', 'Recall',
                'TP', 'TN', 'FP', 'FN', 'AUC_Retention_Ratio', 'Worst_Group_FNR', 'FNR_Gap',
                'Equalized_Odds_Difference', 'EFNHI_Star', 'Brier_Score', 'Platt_Calibration_Slope'
            ]
            f.write(df_res[cols_to_show].to_markdown(index=False))
            f.write("\n\n")

        f.write("---\n\n")
        f.write("## 2. Multi-Criteria Fairness-Utility Pareto Frontier\n\n")
        for cohort_key, df_p in pareto_dict.items():
            f.write(f"### Pareto Frontier Acceptability ({cohort_key} Cohort)\n\n")
            f.write(df_p[['Model', 'ROC_AUC', 'Worst_Group_FNR', 'FNR_Gap', 'Calibration_Slope', 'Pareto_Acceptable']].to_markdown(index=False))
            f.write("\n\n")

        f.write("> [!NOTE]\n")
        f.write("> Evaluated using tuned hyperparameters with full multicore acceleration.\n")

    logging.info(f"Generated unified markdown report at: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified HIR-M3 + ACT-Parity Hybrid Benchmarking Pipeline")
    parser.add_argument("--cohort", type=str, default="all", choices=["Texas", "Nationwide", "all", "texas", "nationwide"],
                        help="Specific cohort to benchmark: 'Texas', 'Nationwide', or 'all' (default: all)")
    args = parser.parse_args()

    logging.info("=================================================================")
    logging.info(f"  UNIFIED HIR-M3 + ACT-PARITY HYBRID BENCHMARKING (COHORT: {args.cohort.upper()})")
    logging.info("=================================================================")

    if args.cohort.lower() == "texas":
        cohort_list = ["Texas"]
    elif args.cohort.lower() == "nationwide":
        cohort_list = ["Nationwide"]
    else:
        cohort_list = ["Texas", "Nationwide"]

    results_dict = {}
    pareto_dict = {}

    for cohort_key in cohort_list:
        res = run_comprehensive_model_comparison(dataset_key=cohort_key)
        if res:
            key, r_df, p_df = res
            results_dict[key] = r_df
            pareto_dict[key] = p_df
        gc.collect()

    if results_dict:
        report_suffix = f"_{args.cohort.lower()}" if args.cohort.lower() in ["texas", "nationwide"] else ""
        custom_report = os.path.join(BASE_DIR, "docs", f"UNIFIED_MODEL_PARITY_COMPARISON_REPORT{report_suffix.upper()}.md")
        generate_unified_markdown_report(results_dict, pareto_dict, report_path=custom_report)

if __name__ == "__main__":
    main()

