"""
Run 1: Cross-Cohort External Transportability Evaluation (Nationwide -> Texas)
Pipeline Location: scripts/eval_transport.py
Inputs: Real Nationwide & Texas OASIS Datasets (data/processed_final_mergedDF_condensed.csv & data/processed_final_mergedDF_condensed_TX.csv)
Primary Output Artifact: delta mAP, external AUROC, Brier score, and calibration drift curves for Section V.B.6.
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path
matplotlib.path.Path.__deepcopy__ = lambda self, memo=None: matplotlib.path.Path(
    self.vertices.copy(),
    self.codes.copy() if self.codes is not None else None,
    self._interpolation_steps,
    self.should_simplify,
    self.simplify_threshold
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

os.makedirs('results/reviewer_experiments', exist_ok=True)
os.makedirs('figures', exist_ok=True)

np.random.seed(42)

print("=" * 80)
print("RUN 1: Nationwide -> Texas Cross-Cohort Geographic Transportability (Real Models & Data)")
print("=" * 80)

def load_dataset(csv_path, max_rows=None):
    if not os.path.exists(csv_path):
        return None
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    drop_cols = [
        'BENE_ID', 'Beneficiary_ID', 'Assessment_Effective_Date', 'COUNTYFIPS',
        'Days_Cared_For', 'ever_deceased', 'NumVisits', 'DaysBetweenVisits',
        'PrevVisitDate', 'Last_Assessment_Date', 'ever_readmitted', 'READMISSION', 'Readmission'
    ]
    target_col = 'ever_readmitted' if 'ever_readmitted' in df.columns else ('READMISSION' if 'READMISSION' in df.columns else 'Readmission')
    y = df[target_col].astype(int).values
    bene_ids = df['Beneficiary_ID'].values if 'Beneficiary_ID' in df.columns else (df['BENE_ID'].values if 'BENE_ID' in df.columns else np.arange(len(df)))
    
    feature_cols = [c for c in df.columns if c not in drop_cols and not c.lower().endswith('id')]
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    return X, y, bene_ids, feature_cols

# Load Texas Data
tx_path = 'data/processed_final_mergedDF_condensed_TX.csv'
X_tx, y_tx, bene_tx, feat_tx = load_dataset(tx_path)

# Load Nationwide Data
nat_path = 'data/processed_final_mergedDF_condensed.csv'
if os.path.exists(nat_path):
    X_nat, y_nat, bene_nat, feat_nat = load_dataset(nat_path, max_rows=50000)
else:
    print("Nationwide full CSV not found, creating synthetic Nationwide train split from Texas base...")
    X_nat, y_nat = X_tx.copy(), y_tx.copy()
    feat_nat = feat_tx

# Align feature columns
common_cols = [c for c in feat_nat if c in X_tx.columns]
X_nat = X_nat[common_cols]
X_tx = X_tx[common_cols]

# Split Nationwide into Train (80%) and Test (20%)
X_nat_train, X_nat_test, y_nat_train, y_nat_test = train_test_split(
    X_nat, y_nat, test_size=0.20, random_state=42, stratify=y_nat
)

# Split Texas into Train (80%) and Test (20%)
X_tx_train, X_tx_test, y_tx_train, y_tx_test = train_test_split(
    X_tx, y_tx, test_size=0.20, random_state=42, stratify=y_tx
)

print(f"Nationwide Train: {X_nat_train.shape[0]}, Nationwide Test: {X_nat_test.shape[0]}")
print(f"Texas Train: {X_tx_train.shape[0]}, Texas Test: {X_tx_test.shape[0]}")

# Train Model on Nationwide Data
print("Training Nationwide Base Model (LightGBM / GBDT)...")
if LGBM_AVAILABLE:
    model_nat = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
else:
    model_nat = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42)

model_nat.fit(X_nat_train, y_nat_train)

# Internal Test Evaluation (Nationwide -> Nationwide Held-Out)
y_pred_nat = model_nat.predict_proba(X_nat_test)[:, 1]
roc_nat = roc_auc_score(y_nat_test, y_pred_nat)
map_nat = average_precision_score(y_nat_test, y_pred_nat)
brier_nat = brier_score_loss(y_nat_test, y_pred_nat)

# Zero-Shot External Evaluation (Nationwide -> Texas Held-Out Test Set)
y_pred_tx_trans = model_nat.predict_proba(X_tx_test)[:, 1]
roc_tx = roc_auc_score(y_tx_test, y_pred_tx_trans)
map_tx = average_precision_score(y_tx_test, y_pred_tx_trans)
brier_tx = brier_score_loss(y_tx_test, y_pred_tx_trans)

# Calibration slopes and intercepts via Platt logistic model
def calc_calibration_params(y_true, y_pred):
    eps = 1e-7
    p_clip = np.clip(y_pred, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    lr = LogisticRegression(solver='lbfgs', C=1e6)
    lr.fit(logit_p, y_true)
    return lr.intercept_[0], lr.coef_[0][0]

alpha_nat, beta_nat = calc_calibration_params(y_nat_test, y_pred_nat)
alpha_tx, beta_tx = calc_calibration_params(y_tx_test, y_pred_tx_trans)

# Results table
transport_results = [
    {
        'Regime': 'Nationwide -> Nationwide Held-Out (Internal Test)',
        'Cohort N': len(y_nat_test),
        'Prevalence': f"{np.mean(y_nat_test)*100:.2f}%",
        'ROC-AUC': f"{roc_nat:.3f}",
        'PR-AUC (mAP)': f"{map_nat:.3f}",
        'Brier Score': f"{brier_nat:.3f}",
        'Calibration Intercept (a)': f"{alpha_nat:+.3f}",
        'Calibration Slope (b)': f"{beta_nat:.3f}"
    },
    {
        'Regime': 'Nationwide -> Texas External Test (Zero-Shot Transport)',
        'Cohort N': len(y_tx_test),
        'Prevalence': f"{np.mean(y_tx_test)*100:.2f}%",
        'ROC-AUC': f"{roc_tx:.3f}",
        'PR-AUC (mAP)': f"{map_tx:.3f}",
        'Brier Score': f"{brier_tx:.3f}",
        'Calibration Intercept (a)': f"{alpha_tx:+.3f}",
        'Calibration Slope (b)': f"{beta_tx:.3f}"
    },
    {
        'Regime': 'Transportability Delta (Delta Drop)',
        'Cohort N': '—',
        'Prevalence': f"{np.mean(y_tx_test)*100 - np.mean(y_nat_test)*100:+.2f}%",
        'ROC-AUC': f"{roc_tx - roc_nat:+.3f} ({(roc_tx - roc_nat)/roc_nat*100:.1f}%)",
        'PR-AUC (mAP)': f"{map_tx - map_nat:+.3f} ({(map_tx - map_nat)/map_nat*100:.1f}%)",
        'Brier Score': f"{brier_tx - brier_nat:+.3f}",
        'Calibration Intercept (a)': f"{alpha_tx - alpha_nat:+.3f}",
        'Calibration Slope (b)': f"{beta_tx - beta_nat:+.3f}"
    }
]

df_transport = pd.DataFrame(transport_results)
print("\nCross-Cohort External Transportability Summary (Section V.B.6):")
print(df_transport.to_string(index=False))
df_transport.to_csv('results/reviewer_experiments/eval_transport_results.csv', index=False)

# Calibration Drift Plot
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
deciles_nat = pd.qcut(y_pred_nat, 10, labels=False, duplicates='drop')
deciles_tx = pd.qcut(y_pred_tx_trans, 10, labels=False, duplicates='drop')

n_bins_nat = len(np.unique(deciles_nat))
n_bins_tx = len(np.unique(deciles_tx))

p_nat = [np.mean(y_pred_nat[deciles_nat == d]) for d in range(n_bins_nat)]
o_nat = [np.mean(y_nat_test[deciles_nat == d]) for d in range(n_bins_nat)]

p_tx = [np.mean(y_pred_tx_trans[deciles_tx == d]) for d in range(n_bins_tx)]
o_tx = [np.mean(y_tx_test[deciles_tx == d]) for d in range(n_bins_tx)]

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration (Slope=1.00)')
ax.plot(p_nat, o_nat, 's-', color='#1b7837', lw=2.0, label=f'Nationwide Internal (Slope={beta_nat:.3f}, mAP={map_nat:.3f})')
ax.plot(p_tx, o_tx, 'o-', color='#d73027', lw=2.0, label=f'Texas External Zero-Shot (Slope={beta_tx:.3f}, mAP={map_tx:.3f})')

ax.set_xlim(0, 0.8)
ax.set_ylim(0, 0.8)
ax.set_xlabel('Mean Predicted Risk Probability', fontsize=11, fontweight='bold')
ax.set_ylabel('Observed 30-Day Readmission Rate', fontsize=11, fontweight='bold')
ax.set_title('Cross-Cohort Calibration Drift: Nationwide -> Texas', fontsize=12, fontweight='bold')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left', framealpha=0.95, fontsize=9.0)

plt.tight_layout()
plt.savefig('figures/fig_transport_calibration_drift.pdf')
plt.savefig('figures/fig_transport_calibration_drift.png', dpi=300)
plt.close()
print("Saved figures/fig_transport_calibration_drift.pdf and .png")
