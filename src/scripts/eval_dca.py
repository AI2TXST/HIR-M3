"""
Run 3: Decision Curve Analysis (DCA Net Benefit Across Exchange Rates pt in [0.01, 0.50])
Pipeline Location: scripts/eval_dca.py
Inputs: Real Texas Patient Cohort Features (data/processed_final_mergedDF_condensed_TX.csv)
Primary Output Artifact: Publication-quality Net Benefit Decision Curve (Figure 2, PDF & PNG)
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

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

os.makedirs('results/reviewer_experiments', exist_ok=True)
os.makedirs('figures', exist_ok=True)
np.random.seed(42)

print("=" * 80)
print("RUN 3: Decision Curve Analysis (DCA) Across Exchange Rates pt in [0.01, 0.50] (Real Models)")
print("=" * 80)

# Load real Texas patient data
tx_csv_path = 'data/processed_final_mergedDF_condensed_TX.csv'
if not os.path.exists(tx_csv_path):
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

feature_cols = [c for c in df_tx.columns if c not in drop_cols and not c.lower().endswith('id')]
X_all = df_tx[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

# Stratified Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)

# Train Models
print("Training Standalone GBDT / LightGBM...")
if LGBM_AVAILABLE:
    model_gbdt = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
else:
    model_gbdt = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42)
model_gbdt.fit(X_train, y_train)

print("Training Secondary Model (Random Forest / Logistic Regression)...")
model_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)

# Generate Predictions on Held-Out Test Set
p_gbdt = model_gbdt.predict_proba(X_test)[:, 1]
p_rf = model_rf.predict_proba(X_test)[:, 1]

# Lead Ensemble (70% GBDT : 30% RF / Deep Tabular)
p_ens = 0.70 * p_gbdt + 0.30 * p_rf

y_true = y_test
n_total = len(y_true)
prevalence = np.mean(y_true)

# DCA Net Benefit Formula
def calc_net_benefit(y_t, y_p, pt_grid):
    nb = []
    n = len(y_t)
    for pt in pt_grid:
        pred_pos = (y_p >= pt)
        tp = np.sum((pred_pos == 1) & (y_t == 1))
        fp = np.sum((pred_pos == 1) & (y_t == 0))
        weight = pt / (1.0 - pt)
        net_b = (tp / n) - (fp / n) * weight
        nb.append(net_b)
    return np.array(nb)

pt_range = np.linspace(0.01, 0.50, 100)

nb_treat_all = prevalence - (1.0 - prevalence) * (pt_range / (1.0 - pt_range))
nb_treat_none = np.zeros_like(pt_range)
nb_ens = calc_net_benefit(y_true, p_ens, pt_range)
nb_gbdt = calc_net_benefit(y_true, p_gbdt, pt_range)
nb_rf = calc_net_benefit(y_true, p_rf, pt_range)

# Export DCA Table
df_dca_export = pd.DataFrame({
    'Threshold_pt': pt_range,
    'Net_Benefit_Lead_Ensemble': nb_ens,
    'Net_Benefit_GBDT': nb_gbdt,
    'Net_Benefit_RandomForest': nb_rf,
    'Net_Benefit_Treat_All': nb_treat_all,
    'Net_Benefit_Treat_None': nb_treat_none
})
df_dca_export.to_csv('results/reviewer_experiments/eval_dca_results.csv', index=False)

# Figure 2: Publication Quality Vector Graphic
fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=300)

ax.plot(pt_range, nb_treat_none, 'k-', lw=1.5, label='Treat None (Net Benefit = 0)')
ax.plot(pt_range, nb_treat_all, 'k--', lw=1.5, label='Treat All (Universal Outreach)')
ax.plot(pt_range, nb_ens, '-', color='#1b7837', lw=2.5, label='Lead Ensemble (70% GBDT : 30% Partner)')
ax.plot(pt_range, nb_gbdt, ':', color='#d73027', lw=1.8, label='Standalone GBDT')
ax.plot(pt_range, nb_rf, '-.', color='#2166ac', lw=1.8, label='Standalone Random Forest')

# Highlight clinical actionability window
ax.axvspan(0.05, 0.25, color='#e8f5e9', alpha=0.5, label='Clinically Actionable Window (pt in [0.05, 0.25])')

# Peak Net Benefit Annotation
idx_15 = np.argmin(np.abs(pt_range - 0.15))
val_15 = nb_ens[idx_15]
ax.annotate(f'Lead Ensemble Net Benefit = +{val_15:.4f}\nat pt = 0.15',
            xy=(0.15, val_15), xytext=(0.24, val_15 + 0.030),
            arrowprops=dict(arrowstyle="->", lw=1.5, color='#1b7837'),
            fontsize=9.0, fontweight='bold', color='#1b7837',
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#1b7837", lw=1.2))

ax.set_ylim(-0.03, max(nb_ens) * 1.25)
ax.set_xlim(0.01, 0.50)
ax.set_xlabel('Decision Threshold Probability (pt)', fontsize=11, fontweight='bold')
ax.set_ylabel('Net Clinical Benefit', fontsize=11, fontweight='bold')
ax.set_title('Figure 2: Clinical Decision Curve Analysis (DCA)\nTransitional Care Management Net Benefit Across Decision Thresholds', fontsize=12, fontweight='bold', pad=12)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right', framealpha=0.95, fontsize=8.5)

plt.tight_layout()
plt.savefig('figures/fig2_dca_net_benefit.pdf')
plt.savefig('figures/fig2_dca_net_benefit.png', dpi=300)
plt.close()
print("Saved figures/fig2_dca_net_benefit.pdf and .png")
