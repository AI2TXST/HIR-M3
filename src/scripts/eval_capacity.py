"""
Run 2: Alert Volume Capacity Audit (Top 5%, 10%, 15% Operational Slices)
Pipeline Location: scripts/eval_capacity.py
Inputs: Real Texas Patient Cohort Features (data/processed_final_mergedDF_condensed_TX.csv)
Primary Output Artifact: Capacity-Alert Volume Table with 1,000 Cluster-Bootstrap CIs for Section V
"""

import os
import sys
import gc
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

os.makedirs('results/reviewer_experiments', exist_ok=True)
np.random.seed(42)

print("=" * 80)
print("RUN 2: Alert Volume Capacity Audit (Top 5%, 10%, 15% Operational Slices - Real Models)")
print("=" * 80)

# Load actual Texas Patient Data
tx_csv_path = 'data/processed_final_mergedDF_condensed_TX.csv'
if not os.path.exists(tx_csv_path):
    # Search fallback locations
    candidates = ['../data/processed_final_mergedDF_condensed_TX.csv', 'data/processed_final_mergedDF_TX.csv']
    tx_csv_path = next((c for c in candidates if os.path.exists(c)), tx_csv_path)

print(f"Loading patient cohort from {tx_csv_path}...")
df_tx = pd.read_csv(tx_csv_path, low_memory=False)

drop_cols = [
    'BENE_ID', 'Beneficiary_ID', 'Assessment_Effective_Date', 'COUNTYFIPS',
    'Days_Cared_For', 'ever_deceased', 'NumVisits', 'DaysBetweenVisits',
    'PrevVisitDate', 'Last_Assessment_Date', 'ever_readmitted', 'READMISSION', 'Readmission'
]
target_col = 'ever_readmitted' if 'ever_readmitted' in df_tx.columns else ('READMISSION' if 'READMISSION' in df_tx.columns else 'Readmission')
y_all = df_tx[target_col].astype(int).values
bene_ids_all = df_tx['Beneficiary_ID'].values if 'Beneficiary_ID' in df_tx.columns else (df_tx['BENE_ID'].values if 'BENE_ID' in df_tx.columns else np.arange(len(df_tx)))

feature_cols = [c for c in df_tx.columns if c not in drop_cols and not c.lower().endswith('id')]
X_all = df_tx[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

print(f"Cohort dimensions: {X_all.shape}, Event Rate: {np.mean(y_all)*100:.2f}%")

# Train/Test Split (80% Train, 20% Held-Out Test)
X_train, X_test, y_train, y_test, bene_train, bene_test = train_test_split(
    X_all, y_all, bene_ids_all, test_size=0.20, random_state=42, stratify=y_all
)

print(f"Training Lead Ensemble / GBDT on {len(X_train)} patients...")
if LGBM_AVAILABLE:
    model = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
else:
    model = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42)

model.fit(X_train, y_train)

# Predict risk probabilities on held-out test set
y_pred = model.predict_proba(X_test)[:, 1]
y_true = y_test
n_test = len(y_true)
n_pos = int(np.sum(y_true))
n_neg = n_test - n_pos

print(f"Held-Out Test Size: {n_test} patients, {n_pos} readmissions ({n_pos/n_test*100:.2f}%)")

# Sort descending by risk score
sorted_indices = np.argsort(y_pred)[::-1]
y_sorted = y_true[sorted_indices]
scores_sorted = y_pred[sorted_indices]
bene_sorted = bene_test[sorted_indices]

capacity_slices = [0.05, 0.10, 0.15]
capacity_rows = []

# 1,000 Bootstrap iterations for confidence intervals
n_boot = 1000
unique_benes = np.unique(bene_test)
print(f"Running {n_boot} cluster-bootstrap iterations on Beneficiary_ID...")

# Pre-generate bootstrap samples of beneficiaries
boot_indices_pool = []
bene_to_idx = {b: np.where(bene_test == b)[0] for b in unique_benes}

for _ in range(n_boot):
    sample_benes = np.random.choice(unique_benes, size=len(unique_benes), replace=True)
    sample_idx = np.concatenate([bene_to_idx[b] for b in sample_benes])
    boot_indices_pool.append(sample_idx)

for cap in capacity_slices:
    k = int(n_test * cap)
    cutoff = scores_sorted[k - 1]
    
    intervened = y_sorted[:k]
    tp = np.sum(intervened == 1)
    fp = k - tp
    fn = n_pos - tp
    tn = n_neg - fp
    
    sens_pt = tp / n_pos if n_pos > 0 else 0
    spec_pt = tn / n_neg if n_neg > 0 else 0
    ppv_pt = tp / k if k > 0 else 0
    nns_pt = 1.0 / ppv_pt if ppv_pt > 0 else np.nan
    reads_per_1000 = (tp / k) * 1000 if k > 0 else 0
    
    # Bootstrap CIs
    sens_boot, spec_boot, ppv_boot = [], [], []
    for b_idx in boot_indices_pool:
        b_y = y_true[b_idx]
        b_p = y_pred[b_idx]
        b_pos = np.sum(b_y == 1)
        b_neg = len(b_y) - b_pos
        
        # Apply the fixed score cutoff
        b_pred_pos = (b_p >= cutoff)
        b_tp = np.sum((b_pred_pos == 1) & (b_y == 1))
        b_fp = np.sum((b_pred_pos == 1) & (b_y == 0))
        b_fn = b_pos - b_tp
        b_tn = b_neg - b_fp
        
        sens_b = b_tp / b_pos if b_pos > 0 else 0
        spec_b = b_tn / b_neg if b_neg > 0 else 0
        ppv_b = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0
        
        sens_boot.append(sens_b)
        spec_boot.append(spec_b)
        ppv_boot.append(ppv_b)
        
    sens_ci = np.percentile(sens_boot, [2.5, 97.5])
    spec_ci = np.percentile(spec_boot, [2.5, 97.5])
    ppv_ci = np.percentile(ppv_boot, [2.5, 97.5])
    
    capacity_rows.append({
        'Capacity Tier': f"Top {int(cap*100)}% Alert Volume",
        'Risk Cutoff Threshold': f"p_hat >= {cutoff:.3f}",
        'Patients Flagged (k)': k,
        'Sensitivity (Recall) [95% CI]': f"{sens_pt*100:.2f}% [{sens_ci[0]*100:.1f}%, {sens_ci[1]*100:.1f}%]",
        'Specificity [95% CI]': f"{spec_pt*100:.2f}% [{spec_ci[0]*100:.1f}%, {spec_ci[1]*100:.1f}%]",
        'PPV (Precision) [95% CI]': f"{ppv_pt*100:.2f}% [{ppv_ci[0]*100:.1f}%, {ppv_ci[1]*100:.1f}%]",
        'Number Needed to Screen (NNS)': f"{nns_pt:.1f}",
        'Readmissions Captured / 1,000 Pts': f"{reads_per_1000:.1f}"
    })

df_capacity = pd.DataFrame(capacity_rows)
print("\nCapacity-Alert Volume Table (Supplementary Table S10 / Table X):")
print(df_capacity.to_string(index=False))

df_capacity.to_csv('results/reviewer_experiments/eval_capacity_results.csv', index=False)
print("\nExported results/reviewer_experiments/eval_capacity_results.csv")
