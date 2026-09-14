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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    precision_score, recall_score, brier_score_loss, confusion_matrix,
    precision_recall_curve
)

# Ensure parity and modeling directories are in sys.path
PARITY_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PARITY_DIR)
MODELING_DIR = os.path.join(BASE_DIR, "modeling")

for d in [BASE_DIR, PARITY_DIR, MODELING_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from parity.models import get_m3_feature_groups_tokenized, ACTParityV2, HIRM3_ACTParity_Hybrid, TORCH_AVAILABLE, DEVICE
from parity.loss import HIRM3_ACTParity_HybridLoss, DualMultiplierManager
from parity.metrics import (
    calculate_stabilized_ctdi,
    calculate_harm_weighted_efnhi,
    calculate_platt_calibration_slope,
    calculate_subgroup_disparity_metrics,
    calculate_bootstrap_confidence_intervals
)

try:
    from modeling.models import (
        train_lgb, train_xgb, train_cb, train_rf, train_logreg, train_hir,
        LGBM_AVAILABLE, XGB_AVAILABLE, CATBOOST_AVAILABLE
    )
except ImportError:
    from models import (
        train_lgb, train_xgb, train_cb, train_rf, train_logreg, train_hir,
        LGBM_AVAILABLE, XGB_AVAILABLE, CATBOOST_AVAILABLE
    )

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Set reproducible random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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
    "Overall": lambda d, hf, hyp: np.ones_like(d, dtype=bool),
    "Urban": lambda d, hf, hyp: np.ones_like(d, dtype=bool),
    "Rural": lambda d, hf, hyp: np.ones_like(d, dtype=bool),
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

def extract_demographic_race_labels(df):
    """Constructs categorical race/ethnicity series for equity disparity metrics."""
    race_series = pd.Series("White", index=df.index)
    demo_cols = {
        'Black_or_African_American': 'Black',
        'Hispanic_or_Latino': 'Hispanic',
        'Asian': 'Asian',
        'American_Indian_or_Alaska_Native': 'AIAN',
        'Native_Hawiian_or_Pacific_Islander': 'NHPI',
        'White': 'White'
    }
    for col, label in demo_cols.items():
        if col in df.columns:
            race_series[df[col] == 1] = label

    if 'race' in df.columns:
        return df['race'].astype(str)
    return race_series.values

def load_and_prep_subcohort_data(filepath, subcohort_name, subgroup_filter_func, target_col=TARGET_COL):
    """Loads dataset, applies sub-cohort filter, extracts features and demographic race codes."""
    actual_path = next((p for p in filepath if os.path.exists(p)), None) if isinstance(filepath, list) else filepath
    if not actual_path or not os.path.exists(str(actual_path)):
        logging.error(f"Dataset path not found: {filepath}")
        return None

    if str(actual_path).endswith('.parquet'):
        logging.info(f"Loading dataset from Parquet: {actual_path}...")
        df = pd.read_parquet(actual_path)
    else:
        logging.info(f"Loading dataset from CSV: {actual_path}...")
        df = pd.concat([c for c in pd.read_csv(actual_path, chunksize=100000, low_memory=False)], ignore_index=True)

    if target_col not in df.columns:
        possible_targets = [c for c in df.columns if 'target' in c.lower() or 'readmit' in c.lower()]
        if possible_targets: target_col = possible_targets[0]

    ruca_col = next((c for c in ['RUCA', 'ruca', 'urban_rural', 'area_type'] if c in df.columns), None)
    if subcohort_name == "Urban":
        if ruca_col:
            if df[ruca_col].dtype == object or df[ruca_col].dtype == str:
                df = df[df[ruca_col].astype(str).str.upper().str.contains('URBAN')].copy()
            else:
                df = df[df[ruca_col] <= 3].copy()
    elif subcohort_name == "Rural":
        if ruca_col:
            if df[ruca_col].dtype == object or df[ruca_col].dtype == str:
                df = df[df[ruca_col].astype(str).str.upper().str.contains('RURAL')].copy()
            else:
                df = df[df[ruca_col] > 3].copy()
    elif subcohort_name != "Overall":
        diab_col = resolve_condition_column(df, ['has_diabetes', 'diab', 'Endocrine_Diabetes'])
        hf_col = resolve_condition_column(df, ['has_heart_failure', 'chf', 'heartfailure', 'Circulatory_HeartFailure'])
        hyp_col = resolve_condition_column(df, ['has_hypertension', 'hypertension', 'hyp', 'Circulatory_Hypertension'])
        if diab_col and hf_col and hyp_col:
            mask = subgroup_filter_func(df[diab_col], df[hf_col], df[hyp_col])
            df = df[mask].copy()

    if len(df) == 0:
        logging.error(f"Sub-cohort '{subcohort_name}' returned 0 rows.")
        return None

    if "TX" not in str(actual_path) and "texas" not in str(actual_path).lower() and len(df) > 50000:
        df, _ = train_test_split(df, train_size=0.50, stratify=df[target_col], random_state=RANDOM_SEED)

    total_sample_size = len(df)
    race_labels = extract_demographic_race_labels(df)

    drop_cols = [
        target_col, 'ID', 'Patient_ID', 'ZipCode', 'Agency_Medicare_Number',
        'Facility_Internal_ID', 'COUNTY_NAME', 'BENE_ID', 'Beneficiary_ID',
        'Assessment_Effective_Date', 'COUNTYFIPS'
    ]
    id_cols = [c for c in df.columns if 'beneficiary' in c.lower() or 'bene_id' in c.lower() or c.lower().endswith('id')]
    non_numeric = df.select_dtypes(include=['object', 'string', 'category']).columns
    drop_cols = list(set(drop_cols).union(set(id_cols)).union(set(non_numeric)))

    feature_cols = [c for c in df.columns if c not in drop_cols and not c.startswith('Unnamed')]

    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        if X_df[col].dtype == object or X_df[col].dtype == str or X_df[col].dtype.name == 'category':
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

    X_df = X_df.dropna(how='all', axis=1).fillna(X_df.median()).astype(np.float32)
    feature_cols = list(X_df.columns)

    X = X_df.values
    y = df[target_col].values.astype(np.int8)

    X_tr, X_te, y_tr, y_te, race_tr, race_te = train_test_split(
        X, y, race_labels, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    return X_tr_s, X_te_s, y_tr, y_te, race_tr, race_te, feature_cols, total_sample_size

def load_tuned_hyperparameters(cohort_name, output_dir):
    """Loads saved tuned ML and Neural hyperparameter JSON files with robust multi-path resolution."""
    raw_prefix = cohort_name.lower().replace(" ", "_")
    short_prefix = raw_prefix.replace("_cohort", "")
    prefixes = list(dict.fromkeys([raw_prefix, f"{short_prefix}_cohort", short_prefix]))

    search_dirs = [
        output_dir,
        os.path.join(MODELING_DIR, "results", "regular_modeling"),
        os.path.join(MODELING_DIR, "results"),
        os.path.join(PARITY_DIR, "results"),
        os.path.join(PARITY_DIR, "..", "modeling", "results", "regular_modeling"),
        os.path.join("results", "regular_modeling"),
        os.path.join("modeling", "results", "regular_modeling"),
        os.path.join("..", "modeling", "results", "regular_modeling")
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

def calculate_demographic_parity_weights(y_train, race_train):
    """Calculates demographic parity sample weights to balance group representation."""
    weights = np.ones(len(y_train), dtype=np.float32)
    unique_groups = np.unique(race_train)
    total_n = len(y_train)
    n_groups = len(unique_groups)

    for g in unique_groups:
        mask = (race_train == g)
        g_count = np.sum(mask)
        if g_count > 0:
            weights[mask] = total_n / (n_groups * g_count)
    return weights

def train_act_parity_hir_model(X_tr_s, y_tr, race_tr, X_te_s, y_te, feature_cols, neural_hp):
    """Trains HIR-M3 Transformer with ACT-Parity Augmented Lagrangian Loss."""
    if not TORCH_AVAILABLE:
        return None

    import torch
    torch.set_num_threads(2)

    if len(feature_cols) > 200:
        variances = np.var(X_tr_s, axis=0)
        top_200_idx = np.argsort(variances)[::-1][:200]
        sub_feature_cols = [feature_cols[i] for i in top_200_idx]
        X_tr_sub = X_tr_s[:, top_200_idx]
        X_te_sub = X_te_s[:, top_200_idx]
    else:
        sub_feature_cols = feature_cols
        X_tr_sub = X_tr_s
        X_te_sub = X_te_s

    micro_cols, meso_cols, macro_cols = get_m3_feature_groups_tokenized(sub_feature_cols)
    micro_idxs = [sub_feature_cols.index(c) for c in micro_cols if c in sub_feature_cols]
    meso_idxs = [sub_feature_cols.index(c) for c in meso_cols if c in sub_feature_cols]
    macro_idxs = [sub_feature_cols.index(c) for c in macro_cols if c in sub_feature_cols]

    if not micro_idxs: micro_idxs = list(range(min(10, len(sub_feature_cols))))

    X_tr_mic = X_tr_sub[:, micro_idxs]
    X_tr_mes = X_tr_sub[:, meso_idxs] if meso_idxs else np.zeros((len(X_tr_sub), 1), dtype=np.float32)
    X_tr_mac = X_tr_sub[:, macro_idxs] if macro_idxs else np.zeros((len(X_tr_sub), 1), dtype=np.float32)

    X_te_mic = X_te_sub[:, micro_idxs]
    X_te_mes = X_te_sub[:, meso_idxs] if meso_idxs else np.zeros((len(X_te_sub), 1), dtype=np.float32)
    X_te_mac = X_te_sub[:, macro_idxs] if macro_idxs else np.zeros((len(X_te_sub), 1), dtype=np.float32)

    unique_groups = sorted(list(set(race_tr)))
    g_map = {g: i for i, g in enumerate(unique_groups)}
    g_tr_code = np.array([g_map.get(g, 0) for g in race_tr], dtype=np.int64)

    batch_size = max(256, neural_hp.get('BATCH_SIZE', 256))
    epochs = min(3, neural_hp.get('EPOCHS', 3))

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_mic, dtype=torch.float32),
        torch.tensor(X_tr_mes, dtype=torch.float32),
        torch.tensor(X_tr_mac, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
        torch.tensor(g_tr_code, dtype=torch.long)
    )
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

    model = HIRM3_ACTParity_Hybrid(
        num_micro=X_tr_mic.shape[1],
        num_meso=X_tr_mes.shape[1],
        num_macro=X_tr_mac.shape[1],
        embed_dim=neural_hp.get('EMBED_DIM', 32),
        num_heads=neural_hp.get('NUM_HEADS', 4),
        alpha=0.5, dropout=0.1
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=neural_hp.get('LR', 1e-3), weight_decay=1e-4)
    criterion = HIRM3_ACTParity_HybridLoss(
        delta=0.04, rho=1.0, lambda_hir=neural_hp.get('LAMBDA_HIR', 0.05),
        gamma=neural_hp.get('GAMMA', 0.5), lambda_inv=0.1
    )
    multiplier_mgr = DualMultiplierManager(eta=0.05)

    for epoch in range(epochs):
        model.train()
        epoch_constraints = {}
        total_batches = len(tr_loader)
        for b_idx, (bx_mic, bx_mes, bx_mac, by, bg) in enumerate(tr_loader):
            bx_mic, bx_mes, bx_mac = bx_mic.to(DEVICE), bx_mes.to(DEVICE), bx_mac.to(DEVICE)
            by, bg = by.to(DEVICE), bg.to(DEVICE)

            optimizer.zero_grad()
            logits, _, attn_w = model(bx_mic, bx_mes, bx_mac)
            loss, c_dict = criterion(
                logits, by, bg, multiplier_mgr.get_all_lambdas(),
                attn_weights=attn_w, micro_idxs=list(range(X_tr_mic.shape[1])),
                meso_idxs=list(range(X_tr_mes.shape[1]))
            )
            loss.backward()
            optimizer.step()
            for k, v in c_dict.items(): epoch_constraints[k] = v

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == total_batches:
                logging.info(f"    [ACT-Parity HIR-M3] Epoch {epoch+1}/{epochs} - Batch {b_idx+1}/{total_batches} - Loss: {loss.item():.4f}")

        multiplier_mgr.update(epoch_constraints)

    model.eval()
    with torch.no_grad():
        te_mic = torch.tensor(X_te_mic, dtype=torch.float32).to(DEVICE)
        te_mes = torch.tensor(X_te_mes, dtype=torch.float32).to(DEVICE)
        te_mac = torch.tensor(X_te_mac, dtype=torch.float32).to(DEVICE)
        logits, _, _ = model(te_mic, te_mes, te_mac)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()

    return probs

# =============================================================================
# STUDY METRIC 1: INCREMENTAL ENSEMBLE VALUE (IEV)
# =============================================================================
def calculate_incremental_ensemble_value(ensemble_records, output_dir):
    """
    Compares selected ensemble models with both component models (Base model and HIR-M3).
    IEV_metric = Ensemble_metric - max(Base_metric, HIR_M3_metric) (higher is better)
    IEV_Brier = min(Base_Brier, HIR_M3_Brier) - Ensemble_Brier (lower is better)
    IEV_error = min(Base_error, HIR_M3_error) - Ensemble_error (lower is better)
    Exports: incremental_ensemble_value_summary.csv
    """
    iev_rows = []

    # Filter non-degenerate ensembles (exclude 100:0 and 0:100)
    for record in ensemble_records:
        ens_name = record['Ensemble Model']
        ratio_str = record['Ensemble Ratio']

        # Skip degenerate ratios
        if ratio_str in ['100:0', '0:100']:
            logging.warning(f"Skipping degenerate ensemble ratio {ratio_str} for IEV calculation.")
            continue

        cohort = record['Cohort']
        base_name = record['Base Model']
        hir_name = record['HIR Model']
        ens_metrics = record['Ensemble Metrics']
        base_metrics = record['Base Metrics']
        hir_metrics = record['HIR Metrics']

        # Metric classification: Higher is better vs. Lower is better
        higher_better_metrics = ['ROC AUC', 'PR AUC', 'F1 Score', 'Recall', 'Specificity', 'Precision', 'Net Benefit', 'FPSA']
        lower_better_error_metrics = ['ACT-Parity Gap', 'FNR Gap', 'FPR Gap', 'ECE', 'Transportability Gap']

        all_metrics_to_eval = higher_better_metrics + ['Brier Score'] + lower_better_error_metrics

        for m_name in all_metrics_to_eval:
            base_val = base_metrics.get(m_name, np.nan)
            hir_val = hir_metrics.get(m_name, np.nan)
            ens_val = ens_metrics.get(m_name, np.nan)

            if pd.isna(base_val) or pd.isna(hir_val) or pd.isna(ens_val):
                iev_val = np.nan
                direction = "N/A"
                interp = "Calculation Warning: Metric unavailable for one or more components"
            else:
                if m_name in higher_better_metrics:
                    direction = "Higher is better"
                    iev_val = ens_val - max(base_val, hir_val)
                elif m_name == 'Brier Score':
                    direction = "Lower is better"
                    iev_val = min(base_val, hir_val) - ens_val
                else:
                    direction = "Lower is better"
                    iev_val = min(base_val, hir_val) - ens_val

                if iev_val > 1e-4:
                    interp = "Positive IEV: Ensemble improves on both individual components"
                elif abs(iev_val) <= 1e-4:
                    interp = "Zero IEV: Ensemble provides no additional value beyond better component"
                else:
                    interp = "Negative IEV: Ensemble is worse than at least one component"

            iev_rows.append({
                'Cohort': cohort,
                'Ensemble Model': ens_name,
                'Base Model': base_name,
                'HIR Model': hir_name,
                'Ensemble Ratio': ratio_str,
                'Metric': m_name,
                'Base Value': round(base_val, 4) if not pd.isna(base_val) else np.nan,
                'HIR-M3 Value': round(hir_val, 4) if not pd.isna(hir_val) else np.nan,
                'Ensemble Value': round(ens_val, 4) if not pd.isna(ens_val) else np.nan,
                'Incremental Ensemble Value': round(iev_val, 4) if not pd.isna(iev_val) else np.nan,
                'Direction': direction,
                'Interpretation': interp,
                'CI Lower': np.nan,
                'CI Upper': np.nan
            })

    iev_df = pd.DataFrame(iev_rows)
    out_path = os.path.join(output_dir, "incremental_ensemble_value_summary.csv")
    iev_df.to_csv(out_path, index=False)
    logging.info(f" Exported Incremental Ensemble Value Summary -> {out_path}")
    return iev_df

# =============================================================================
# STUDY METRIC 2: FAIRNESS-PERFORMANCE STABILITY AREA (FPSA) & THRESHOLD SWEEP
# =============================================================================
def calculate_fairness_performance_stability_area(
    y_true, y_prob, race_labels, cohort_name, model_name,
    default_thresh=0.50, act_thresh=None,
    threshold_grid=np.arange(0.10, 0.51, 0.01),
    max_parity_gap=0.04, min_recall=0.70, min_precision=0.30,
    max_fnr_gap=0.10, max_fpr_gap=0.10, require_positive_net_benefit=False,
    roc_auc=None, pr_auc=None, brier=None
):
    """
    Evaluates model across configurable threshold grid (0.10 to 0.50, step 0.01) to calculate FPSA:
    FPSA = Acceptable_Threshold_Count / Valid_Threshold_Count
    Exports threshold_sweep_metrics.csv and fairness_performance_stability_summary.csv.
    """
    sweep_rows = []
    acceptable_count = 0
    valid_count = 0

    N = len(y_true)
    unique_races = np.unique(race_labels)

    for th in threshold_grid:
        th = round(float(th), 4)

        # Determine threshold type label
        if act_thresh is not None and abs(th - round(float(act_thresh), 4)) < 0.005:
            th_type = "ACT-Parity"
        elif abs(th - default_thresh) < 0.005:
            th_type = "Default"
        else:
            th_type = "Sweep"

        preds = (y_prob >= th).astype(int)
        pos_count = int(np.sum(preds))
        neg_count = N - pos_count

        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        spec = float(tn / (tn + fp + 1e-8))

        # Net Benefit: (TP/N) - (FP/N) * (p_t / (1 - p_t))
        pt = th
        net_benefit = float((tp / N) - (fp / N) * (pt / (1.0 - pt + 1e-8)))

        # Calculate Subgroup FNR and FPR
        sub_fnrs = []
        sub_fprs = []
        valid_subgroups = True
        exclusion_reason = ""

        for g in unique_races:
            mask = (race_labels == g)
            g_y = y_true[mask]
            g_preds = preds[mask]

            g_pos = np.sum(g_y == 1)
            g_neg = np.sum(g_y == 0)

            if g_pos == 0 or g_neg == 0:
                valid_subgroups = False
                exclusion_reason = f"Insufficient subgroup positive ({g_pos}) or negative ({g_neg}) cases in group '{g}'"
                break

            g_tn, g_fp, g_fn, g_tp = confusion_matrix(g_y, g_preds, labels=[0, 1]).ravel()
            fnr_g = g_fn / (g_tp + g_fn)
            fpr_g = g_fp / (g_fp + g_tn)
            sub_fnrs.append(fnr_g)
            sub_fprs.append(fpr_g)

        if valid_subgroups and len(sub_fnrs) > 0:
            valid_threshold = True
            fnr_gap = float(np.max(sub_fnrs) - np.min(sub_fnrs))
            fpr_gap = float(np.max(sub_fprs) - np.min(sub_fprs))
            pop_fnr = fn / (tp + fn + 1e-8)
            act_parity_gap = float(np.max(sub_fnrs) - pop_fnr)
            exclusion_reason = "None"
        else:
            valid_threshold = False
            fnr_gap = np.nan
            fpr_gap = np.nan
            act_parity_gap = np.nan

        # Acceptability Criteria
        if valid_threshold:
            valid_count += 1
            acceptable = (
                (act_parity_gap <= max_parity_gap) and
                (rec >= min_recall) and
                (prec >= min_precision) and
                (fnr_gap <= max_fnr_gap) and
                (fpr_gap <= max_fpr_gap)
            )
            if require_positive_net_benefit:
                acceptable = acceptable and (net_benefit > 0)
            
            if acceptable:
                acceptable_count += 1
        else:
            acceptable = False

        sweep_rows.append({
            'Cohort': cohort_name,
            'Model': model_name,
            'Threshold': th,
            'Threshold Type': th_type,
            'Accuracy': round(acc, 4),
            'F1 Score': round(f1, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'Specificity': round(spec, 4),
            'ROC AUC': round(roc_auc, 4) if roc_auc is not None else np.nan,
            'PR AUC': round(pr_auc, 4) if pr_auc is not None else np.nan,
            'Brier Score': round(brier, 4) if brier is not None else np.nan,
            'ACT-Parity Gap': round(act_parity_gap, 4) if not pd.isna(act_parity_gap) else np.nan,
            'FNR Gap': round(fnr_gap, 4) if not pd.isna(fnr_gap) else np.nan,
            'FPR Gap': round(fpr_gap, 4) if not pd.isna(fpr_gap) else np.nan,
            'Net Benefit': round(net_benefit, 4),
            'Acceptable Threshold': acceptable,
            'Valid Threshold': valid_threshold,
            'Exclusion Reason': exclusion_reason,
            'Predicted Positive Count': pos_count,
            'Predicted Negative Count': neg_count
        })

    fpsa = float(acceptable_count / valid_count) if valid_count > 0 else np.nan

    summary_row = {
        'Cohort': cohort_name,
        'Model': model_name,
        'Threshold Range Start': float(threshold_grid[0]),
        'Threshold Range End': float(threshold_grid[-1]),
        'Threshold Increment': 0.01,
        'Valid Threshold Count': valid_count,
        'Acceptable Threshold Count': acceptable_count,
        'Fairness-Performance Stability Area': round(fpsa, 4) if not pd.isna(fpsa) else np.nan,
        'Maximum Allowed ACT-Parity Gap': max_parity_gap,
        'Minimum Recall': min_recall,
        'Minimum Precision': min_precision,
        'Maximum FNR Gap': max_fnr_gap,
        'Maximum FPR Gap': max_fpr_gap,
        'Positive Net Benefit Required': require_positive_net_benefit,
        'Default Threshold': default_thresh,
        'ACT-Parity Threshold': round(float(act_thresh), 4) if act_thresh is not None else np.nan
    }

    return sweep_rows, summary_row, fpsa

# =============================================================================
# STUDY METRIC 3: TRANSPORTABILITY GAP
# =============================================================================
def calculate_transportability_gap(all_model_results_df, output_dir):
    """
    Compares performance, calibration, utility, and fairness results
    for the same model family across Nationwide and Texas cohorts.
    Transportability_Gap_M = abs(M_Nationwide - M_Texas)
    Exports: transportability_gap_summary.csv
    """
    nat_df = all_model_results_df[all_model_results_df['Cohort'] == 'Nationwide Cohort']
    tx_df = all_model_results_df[all_model_results_df['Cohort'] == 'Texas Cohort']

    metrics_list = [
        ('ROC_AUC', 'Higher is better'),
        ('PR_AUC', 'Higher is better'),
        ('F1_Score', 'Higher is better'),
        ('Brier_Score', 'Lower is better'),
        ('Precision', 'Higher is better'),
        ('Recall', 'Higher is better'),
        ('Specificity', 'Higher is better'),
        ('FNR_Gap_DGAP', 'Lower is better'),
        ('Equalized_Odds_Difference_EOD', 'Lower is better'),
        ('Fairness-Performance Stability Area', 'Higher is better')
    ]

    # Map standardized model family names across cohorts
    model_family_mappings = [
        ("XGBoost baseline", "XGBoost baseline", "XGBoost baseline", "N/A", "N/A"),
        ("HIR-M3 Transformer", "HIR-M3 Transformer", "HIR-M3 Transformer", "N/A", "N/A"),
        ("XGBoost + HIR-M3 Ensemble", "XGBoost + HIR-M3 Ensemble (90:10)", "XGBoost + HIR-M3 Ensemble (30:70)", "90:10", "30:70"),
        ("XGBoost with ACT-parity", "XGBoost with ACT-parity", "XGBoost with ACT-parity", "N/A", "N/A"),
        ("HIR-M3 Transformer with ACT-parity", "HIR-M3 Transformer with ACT-parity", "HIR-M3 Transformer with ACT-parity", "N/A", "N/A"),
        ("XGBoost + HIR-M3 Ensemble with ACT-parity", "XGBoost + HIR-M3 Ensemble (90:10) with ACT-parity", "XGBoost + HIR-M3 Ensemble (30:70) with ACT-parity", "90:10", "30:70"),
        ("LightGBM / CatBoost baseline", "LightGBM baseline", "CatBoost baseline", "N/A", "N/A"),
        ("LightGBM / CatBoost with ACT-parity", "LightGBM with ACT-parity", "CatBoost with ACT-parity", "N/A", "N/A")
    ]

    trans_rows = []

    for std_family, nat_m_name, tx_m_name, nat_ratio, tx_ratio in model_family_mappings:
        nat_row = nat_df[nat_df['Model_Name'] == nat_m_name]
        tx_row = tx_df[tx_df['Model_Name'] == tx_m_name]

        # Check if direct comparison is available
        is_comparable = (len(nat_row) > 0) and (len(tx_row) > 0) and ("LightGBM / CatBoost" not in std_family)

        for m_col, direction in metrics_list:
            nat_val = float(nat_row[m_col].values[0]) if len(nat_row) > 0 and m_col in nat_row.columns else np.nan
            tx_val = float(tx_row[m_col].values[0]) if len(tx_row) > 0 and m_col in tx_row.columns else np.nan

            if is_comparable and not pd.isna(nat_val) and not pd.isna(tx_val):
                gap = abs(nat_val - tx_val)
                if direction == "Higher is better":
                    worst_val = min(nat_val, tx_val)
                else:
                    worst_val = max(nat_val, tx_val)
                notes = "Matched cross-cohort family comparison"
            else:
                gap = np.nan
                worst_val = np.nan
                notes = "Comparison Unavailable: Unmatched model pair or cohort-specific algorithm"

            trans_rows.append({
                'Standardized Model Name': std_family,
                'Nationwide Model Name': nat_m_name,
                'Texas Model Name': tx_m_name,
                'Nationwide Ensemble Ratio': nat_ratio,
                'Texas Ensemble Ratio': tx_ratio,
                'Metric': m_col,
                'Direction': direction,
                'Nationwide Value': round(nat_val, 4) if not pd.isna(nat_val) else np.nan,
                'Texas Value': round(tx_val, 4) if not pd.isna(tx_val) else np.nan,
                'Transportability Gap': round(gap, 4) if not pd.isna(gap) else np.nan,
                'Worst-Cohort Value': round(worst_val, 4) if not pd.isna(worst_val) else np.nan,
                'Comparison Available': is_comparable,
                'Notes': notes
            })

    trans_df = pd.DataFrame(trans_rows)
    out_path = os.path.join(output_dir, "transportability_gap_summary.csv")
    trans_df.to_csv(out_path, index=False)
    logging.info(f" Exported Transportability Gap Summary -> {out_path}")
    return trans_df

# =============================================================================
# FIGURE GENERATION: THRESHOLD SWEEP FIGURES
# =============================================================================
def generate_threshold_sweep_figures(sweep_df, output_dir):
    """
    Generates 7-panel threshold sweep figures for Nationwide and Texas cohorts across models.
    """
    cohorts = sweep_df['Cohort'].unique()
    metrics_to_plot = ['ACT-Parity Gap', 'FNR Gap', 'FPR Gap', 'Recall', 'Precision', 'Specificity', 'Net Benefit']

    for cohort in cohorts:
        c_df = sweep_df[sweep_df['Cohort'] == cohort]
        models = c_df['Model'].unique()

        fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
        fig.suptitle(f"Threshold Sweep Acceptability & Metric Trajectories ({cohort})", fontsize=14, fontweight='bold')

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            for m_name in models:
                m_data = c_df[c_df['Model'] == m_name].sort_values('Threshold')
                ax.plot(m_data['Threshold'], m_data[metric], label=m_name, linewidth=1.8)

            ax.set_ylabel(metric, fontsize=10, fontweight='semibold')
            ax.grid(True, linestyle='--', alpha=0.5)

            # Add reference threshold lines
            if metric == 'ACT-Parity Gap': ax.axhline(0.04, color='red', linestyle=':', label='Max ACT-Parity Gap (0.04)')
            elif metric == 'Recall': ax.axhline(0.70, color='green', linestyle=':', label='Min Recall (0.70)')
            elif metric == 'Precision': ax.axhline(0.30, color='blue', linestyle=':', label='Min Precision (0.30)')
            elif metric == 'FNR Gap': ax.axhline(0.10, color='purple', linestyle=':', label='Max FNR Gap (0.10)')
            elif metric == 'FPR Gap': ax.axhline(0.10, color='orange', linestyle=':', label='Max FPR Gap (0.10)')

            ax.axvline(0.50, color='gray', linestyle='--', alpha=0.7, label='Default Thresh (0.50)')

            if idx == 0:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

        axes[-1].set_xlabel("Decision Threshold (tau)", fontsize=11, fontweight='semibold')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"threshold_sweep_{cohort.lower().replace(' ', '_')}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f" Generated Threshold Sweep Figure -> {fig_path}")

# =============================================================================
# MAIN EXPERIMENT EXECUTION PIPELINE
# =============================================================================
def run_parity_subgroup_experiments(selected_cohort="all", max_workers=4):
    logging.info("=========================================================================")
    logging.info("  STARTING NATIONWIDE VS. TEXAS PARITY & DISPARITY EXPERIMENT PIPELINE")
    logging.info("=========================================================================")
    logging.info(f"Parallel Workers: {max_workers}")
    logging.info(f"Evaluated Subgroups: {list(CONDITION_SUBGROUPS.keys())}")
    logging.info(f"Package Availability -> LightGBM: {LGBM_AVAILABLE}, XGBoost: {XGB_AVAILABLE}, CatBoost: {CATBOOST_AVAILABLE}, PyTorch: {TORCH_AVAILABLE}")

    output_dir = os.path.join(PARITY_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    all_sweep_metrics = []
    all_fpsa_summaries = []
    ensemble_records = []

    if selected_cohort.lower().startswith("tx") or selected_cohort.lower().startswith("texas"):
        target_cohorts = {"Texas Cohort": DATASETS["Texas Cohort"]}
    elif selected_cohort.lower().startswith("nat") or selected_cohort.lower().startswith("nationwide"):
        target_cohorts = {"Nationwide Cohort": DATASETS["Nationwide Cohort"]}
    else:
        target_cohorts = DATASETS

    for cohort_name, filepath in target_cohorts.items():
        is_nationwide = "Nationwide" in cohort_name
        ml_hp, neural_hp = load_tuned_hyperparameters(cohort_name, output_dir)

        # Prepare tuned neural configuration for HIR-M3
        hir_cfg = dict(neural_hp.get('HIR-M3 Transformer', {}))
        hir_cfg.setdefault('BATCH_SIZE', 256)
        hir_cfg.setdefault('EMBED_DIM', 32)
        hir_cfg.setdefault('NUM_HEADS', 4)
        hir_cfg.setdefault('HIDDEN_DIM', 128)
        hir_cfg.setdefault('LR', 1e-3)
        hir_cfg.setdefault('EPOCHS', 3)
        hir_cfg.setdefault('LAMBDA_HIR', 0.05)
        hir_cfg.setdefault('GAMMA', 0.5)
        if hir_cfg.get('BATCH_SIZE', 256) < 256: hir_cfg['BATCH_SIZE'] = 256
        if hir_cfg.get('EPOCHS', 3) > 3: hir_cfg['EPOCHS'] = 3

        for subcohort_name, subgroup_filter_func in CONDITION_SUBGROUPS.items():
            full_subcohort_key = f"{cohort_name} - {subcohort_name}"
            logging.info(f"\n=========================================================")
            logging.info(f"   PARITY EVALUATION SUB-COHORT: {full_subcohort_key}")
            logging.info(f"=========================================================")

            data = load_and_prep_subcohort_data(filepath, subcohort_name, subgroup_filter_func)
            if data is None:
                continue

            X_tr_s, X_te_s, y_tr, y_te, race_tr, race_te, feature_cols, sample_size = data
            w_tr_base = np.ones(len(y_tr), dtype=np.float32)
            w_te_base = np.ones(len(y_te), dtype=np.float32)
            w_tr_parity = calculate_demographic_parity_weights(y_tr, race_tr)

            preds = {}

            # 1. GBDT Baseline & Parity Models (Parallel)
            gbdt_tasks = {}
            if is_nationwide:
                lgb_cfg = ml_hp.get('LightGBM', {'learning_rate': 0.1, 'n_estimators': 200})
                xgb_cfg = ml_hp.get('XGBoost', {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200})

                if LGBM_AVAILABLE:
                    gbdt_tasks['LightGBM baseline'] = lambda: train_lgb(X_tr_s, y_tr, w_tr_base, X_te_s, y_te, lgb_cfg)
                    gbdt_tasks['LightGBM with ACT-parity'] = lambda: train_lgb(X_tr_s, y_tr, w_tr_parity, X_te_s, y_te, lgb_cfg)
                if XGB_AVAILABLE:
                    gbdt_tasks['XGBoost baseline'] = lambda: train_xgb(X_tr_s, y_tr, w_tr_base, X_te_s, y_te, xgb_cfg)
                    gbdt_tasks['XGBoost with ACT-parity'] = lambda: train_xgb(X_tr_s, y_tr, w_tr_parity, X_te_s, y_te, xgb_cfg)
            else:
                cb_cfg = ml_hp.get('CatBoost', {'depth': 6, 'iterations': 200, 'learning_rate': 0.1})
                xgb_cfg = ml_hp.get('XGBoost', {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200})

                if CATBOOST_AVAILABLE:
                    gbdt_tasks['CatBoost baseline'] = lambda: train_cb(X_tr_s, y_tr, w_tr_base, X_te_s, y_te, cb_cfg)
                    gbdt_tasks['CatBoost with ACT-parity'] = lambda: train_cb(X_tr_s, y_tr, w_tr_parity, X_te_s, y_te, cb_cfg)
                if XGB_AVAILABLE:
                    gbdt_tasks['XGBoost baseline'] = lambda: train_xgb(X_tr_s, y_tr, w_tr_base, X_te_s, y_te, xgb_cfg)
                    gbdt_tasks['XGBoost with ACT-parity'] = lambda: train_xgb(X_tr_s, y_tr, w_tr_parity, X_te_s, y_te, xgb_cfg)

            def _run_task(task_name, fn):
                t0 = time.time()
                logging.info(f"  [Worker] Starting {task_name}...")
                try:
                    res = fn()
                    dur = time.time() - t0
                    logging.info(f"  [Worker] Finished {task_name} in {dur:.2f}s")
                    return task_name, res
                except Exception as ex:
                    logging.error(f"  [Worker] Failed {task_name}: {ex}\n{traceback.format_exc()}")
                    return task_name, None

            if gbdt_tasks:
                logging.info(f"Executing {len(gbdt_tasks)} GBDT models in parallel (workers={max_workers})...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_run_task, name, fn): name for name, fn in gbdt_tasks.items()}
                    for f in concurrent.futures.as_completed(futures):
                        name, prob = f.result()
                        if prob is not None:
                            preds[name] = prob

            # 2. PyTorch Neural Models (Sequential for MKL RNG thread safety)
            if TORCH_AVAILABLE:
                if len(feature_cols) > 200:
                    var_arr = np.var(X_tr_s, axis=0)
                    top_200 = np.argsort(var_arr)[::-1][:200]
                    tf_feature_cols = [feature_cols[i] for i in top_200]
                    X_tr_tf = X_tr_s[:, top_200]
                    X_te_tf = X_te_s[:, top_200]
                else:
                    tf_feature_cols = feature_cols
                    X_tr_tf = X_tr_s
                    X_te_tf = X_te_s

                if train_hir is not None:
                    _, prob = _run_task('HIR-M3 Transformer', lambda: train_hir(X_tr_tf, y_tr, w_tr_base, X_te_tf, y_te, w_te_base, tf_feature_cols, hir_cfg))
                    if prob is not None:
                        preds['HIR-M3 Transformer'] = prob

                _, prob_parity = _run_task('HIR-M3 Transformer with ACT-parity', lambda: train_act_parity_hir_model(X_tr_tf, y_tr, race_tr, X_te_tf, y_te, tf_feature_cols, hir_cfg))
                if prob_parity is not None:
                    preds['HIR-M3 Transformer with ACT-parity'] = prob_parity

            # Construct Ensembles from trained atomic predictions
            if is_nationwide:
                if 'XGBoost baseline' in preds and 'HIR-M3 Transformer' in preds:
                    logging.info("  Constructing XGBoost + HIR-M3 Ensemble (90:10)...")
                    preds['XGBoost + HIR-M3 Ensemble (90:10)'] = 0.90 * preds['XGBoost baseline'] + 0.10 * preds['HIR-M3 Transformer']
                if 'XGBoost with ACT-parity' in preds and 'HIR-M3 Transformer with ACT-parity' in preds:
                    logging.info("  Constructing XGBoost + HIR-M3 Ensemble (90:10) with ACT-parity...")
                    preds['XGBoost + HIR-M3 Ensemble (90:10) with ACT-parity'] = 0.90 * preds['XGBoost with ACT-parity'] + 0.10 * preds['HIR-M3 Transformer with ACT-parity']
            else:
                if 'XGBoost baseline' in preds and 'HIR-M3 Transformer' in preds:
                    logging.info("  Constructing XGBoost + HIR-M3 Ensemble (30:70)...")
                    preds['XGBoost + HIR-M3 Ensemble (30:70)'] = 0.30 * preds['XGBoost baseline'] + 0.70 * preds['HIR-M3 Transformer']
                if 'XGBoost with ACT-parity' in preds and 'HIR-M3 Transformer with ACT-parity' in preds:
                    logging.info("  Constructing XGBoost + HIR-M3 Ensemble (30:70) with ACT-parity...")
                    preds['XGBoost + HIR-M3 Ensemble (30:70) with ACT-parity'] = 0.30 * preds['XGBoost with ACT-parity'] + 0.70 * preds['HIR-M3 Transformer with ACT-parity']

            sub_results = []
            sub_metrics_cache = {}

            # Evaluate Each Model
            for m_name, y_prob in preds.items():
                if y_prob is None: continue
                y_prob = np.clip(np.asarray(y_prob).ravel(), 0.0, 1.0)

                prec_c, rec_c, th_c = precision_recall_curve(y_te, y_prob)
                f1_c = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-8)
                opt_idx = np.argmax(f1_c)
                opt_th = float(th_c[opt_idx]) if opt_idx < len(th_c) else 0.5

                y_pred = (y_prob >= opt_th).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()

                auc = float(roc_auc_score(y_te, y_prob))
                pr_auc = float(average_precision_score(y_te, y_prob))
                prec = float(precision_score(y_te, y_pred, zero_division=0))
                rec = float(recall_score(y_te, y_pred, zero_division=0))
                f1 = float(f1_score(y_te, y_pred, zero_division=0))
                acc = float(accuracy_score(y_te, y_pred))
                brier = float(brier_score_loss(y_te, y_prob))

                subgroup_df, summary = calculate_subgroup_disparity_metrics(y_te, y_prob, race_te)
                platt_s, _, _ = calculate_platt_calibration_slope(y_te, y_prob)

                # Calculate FPSA & Threshold Sweep for Overall cohort evaluation
                if subcohort_name == "Overall":
                    sw_rows, fpsa_sum, fpsa_val = calculate_fairness_performance_stability_area(
                        y_te, y_prob, race_te, cohort_name, m_name,
                        default_thresh=0.50, act_thresh=opt_th,
                        roc_auc=auc, pr_auc=pr_auc, brier=brier
                    )
                    all_sweep_metrics.extend(sw_rows)
                    all_fpsa_summaries.append(fpsa_sum)
                else:
                    fpsa_val = np.nan

                m_dict = {
                    'ROC AUC': auc, 'PR AUC': pr_auc, 'F1 Score': f1, 'Brier Score': brier,
                    'Precision': prec, 'Recall': rec, 'Specificity': float(tn / (tn + fp + 1e-8)),
                    'ACT-Parity Gap': summary['Delta_FNR'], 'FNR Gap': summary['Delta_FNR'],
                    'FPR Gap': summary.get('Delta_FPR', summary['Delta_FNR']), 'FPSA': fpsa_val
                }
                sub_metrics_cache[m_name] = m_dict

                # Record ensemble component details for IEV calculation
                ratio_str = "N/A"
                if "90:10" in m_name: ratio_str = "90:10"
                elif "30:70" in m_name: ratio_str = "30:70"

                if "Ensemble" in m_name and subcohort_name == "Overall":
                    base_m = "XGBoost baseline" if "XGBoost" in m_name else "CatBoost baseline"
                    hir_m = "HIR-M3 Transformer"
                    if "with ACT-parity" in m_name:
                        base_m = "XGBoost with ACT-parity"
                        hir_m = "HIR-M3 Transformer with ACT-parity"

                    ensemble_records.append({
                        'Cohort': cohort_name,
                        'Ensemble Model': m_name,
                        'Base Model': base_m,
                        'HIR Model': hir_m,
                        'Ensemble Ratio': ratio_str,
                        'Ensemble Metrics': m_dict,
                        'Base Metrics': sub_metrics_cache.get(base_m, m_dict),
                        'HIR Metrics': sub_metrics_cache.get(hir_m, m_dict)
                    })

                res_item = {
                    'Cohort': cohort_name,
                    'Subcohort_Subgroup': subcohort_name,
                    'Model_Name': m_name,
                    'Optimal_Threshold': round(opt_th, 4),
                    'ROC_AUC': round(auc, 4),
                    'PR_AUC': round(pr_auc, 4),
                    'F1_Score': round(f1, 4),
                    'Accuracy': round(acc, 4),
                    'Precision': round(prec, 4),
                    'Recall': round(rec, 4),
                    'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
                    'Overall_FNR': summary['Overall_FNR'],
                    'Worst_Group_FNR': summary['Worst_Group_FNR'],
                    'FNR Gap': summary['Delta_FNR'],
                    'FPR Gap': summary.get('Delta_FPR', summary['Delta_FNR']),
                    'FNR_Gap_DGAP': summary['Delta_FNR'],
                    'Equalized_Odds_Difference_EOD': summary['Equalized_Odds_Difference'],
                    'Demographic_Parity_Ratio_DPR': summary.get('Demographic_Parity_Ratio', 1.0),
                    'Fairness-Performance Stability Area': round(fpsa_val, 4) if not pd.isna(fpsa_val) else np.nan,
                    'Selected Ensemble Ratio': ratio_str,
                    'EFNHI_Star': summary['EFNHI_Star'],
                    'Brier_Score': round(brier, 4),
                    'Platt_Calibration_Slope': round(platt_s, 4)
                }
                all_results.append(res_item)
                sub_results.append(res_item)

            if sub_results:
                clean_name = subcohort_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_plus_")
                out_sub_path = os.path.join(output_dir, f"parity_{cohort_name.replace(' ', '_')}_{clean_name}.csv")
                pd.DataFrame(sub_results).to_csv(out_sub_path, index=False)

            del X_tr_s, X_te_s, y_tr, y_te
            gc.collect()

    # Convert overall main results into DataFrame
    res_df = pd.DataFrame(all_results)

    # 1. Export Threshold Sweep Metrics & FPSA Summary
    if all_sweep_metrics:
        sweep_df = pd.DataFrame(all_sweep_metrics)
        sweep_path = os.path.join(output_dir, "threshold_sweep_metrics.csv")
        sweep_df.to_csv(sweep_path, index=False)
        logging.info(f" Exported Threshold Sweep Metrics -> {sweep_path}")

        generate_threshold_sweep_figures(sweep_df, output_dir)

    if all_fpsa_summaries:
        fpsa_df = pd.DataFrame(all_fpsa_summaries)
        fpsa_path = os.path.join(output_dir, "fairness_performance_stability_summary.csv")
        fpsa_df.to_csv(fpsa_path, index=False)
        logging.info(f" Exported FPSA Summary -> {fpsa_path}")

    # 2. Export Incremental Ensemble Value (IEV) Summary
    if ensemble_records:
        iev_df = calculate_incremental_ensemble_value(ensemble_records, output_dir)
        # Merge IEV metrics back into main results table
        iev_pivot = iev_df.pivot(index=['Cohort', 'Ensemble Model'], columns='Metric', values='Incremental Ensemble Value').reset_index()
        for metric_col in ['ROC AUC', 'PR AUC', 'F1 Score', 'Brier Score', 'ACT-Parity Gap', 'FNR Gap', 'FPR Gap']:
            if metric_col in iev_pivot.columns:
                target_col = f"Incremental Ensemble Value for {metric_col}"
                res_df[target_col] = np.nan
                for _, row in iev_pivot.iterrows():
                    mask = (res_df['Cohort'] == row['Cohort']) & (res_df['Model_Name'] == row['Ensemble Model'])
                    res_df.loc[mask, target_col] = row[metric_col]

    # 3. Export Transportability Gap Summary
    trans_df = calculate_transportability_gap(res_df[res_df['Subcohort_Subgroup'] == 'Overall'], output_dir)
    trans_pivot = trans_df.pivot(index=['Nationwide Model Name'], columns='Metric', values='Transportability Gap').reset_index()

    for m_col, col_name in [
        ('ROC_AUC', 'Transportability Gap for ROC AUC'),
        ('PR_AUC', 'Transportability Gap for PR AUC'),
        ('Brier_Score', 'Transportability Gap for Brier Score'),
        ('FNR_Gap_DGAP', 'Transportability Gap for ACT-Parity Gap'),
        ('FNR_Gap_DGAP', 'Transportability Gap for FNR Gap'),
        ('Equalized_Odds_Difference_EOD', 'Transportability Gap for FPR Gap'),
        ('Fairness-Performance Stability Area', 'Transportability Gap for FPSA')
    ]:
        if m_col in trans_pivot.columns:
            res_df[col_name] = np.nan
            for _, row in trans_pivot.iterrows():
                mask = (res_df['Model_Name'] == row['Nationwide Model Name'])
                res_df.loc[mask, col_name] = row[m_col]

    # Export final updated main parity summary CSV
    final_csv_path = os.path.join(output_dir, "parity_subgroups_modeling_results.csv")
    res_df.to_csv(final_csv_path, index=False)

    # -----------------------------------------------------------------
    # MARKDOWN SUMMARY REPORT GENERATION
    # -----------------------------------------------------------------
    md_report_path = os.path.join(output_dir, "ACT_PARITY_EXTENDED_METRICS_REPORT.md")
    with open(md_report_path, "w") as f:
        f.write("# ACT-Parity Extended Study Metrics Report\n\n")
        f.write("## 1. Incremental Ensemble Value (IEV)\n\n")
        if 'iev_df' in locals() and not iev_df.empty:
            f.write(iev_df.to_markdown(index=False))
        f.write("\n\n## 2. Fairness-Performance Stability Area (FPSA)\n\n")
        if 'fpsa_df' in locals() and not fpsa_df.empty:
            f.write(fpsa_df.to_markdown(index=False))
        f.write("\n\n## 3. Transportability Gap (Nationwide vs. Texas)\n\n")
        if 'trans_df' in locals() and not trans_df.empty:
            f.write(trans_df.to_markdown(index=False))

    logging.info(f"\n=========================================================")
    logging.info(f" Saved consolidated main parity results to: {final_csv_path}")
    logging.info(f" Saved Markdown extended metrics report to: {md_report_path}")
    logging.info(f"=========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nationwide vs Texas Parity Modeling Pipeline")
    parser.add_argument("--cohort", choices=["Texas", "Nationwide", "all"], default="all", help="Cohort selection")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker threads (default: 4)")
    args = parser.parse_args()
    run_parity_subgroup_experiments(args.cohort, max_workers=args.workers)
