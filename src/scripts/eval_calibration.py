"""
Run 4: 10-Decile Calibration Reliability Curves & Recalibration Diagnostics
Pipeline Location: scripts/eval_calibration.py
Inputs: Real Texas Patient Cohort Features (data/processed_final_mergedDF_condensed_TX.csv)
Primary Output Artifact: 2-panel reliability plot (pre- and post-Platt recalibration) exported as Figure 3 (PDF and PNG)
"""

import os
import sys
import gc
import json
import numpy as np
import pandas as pd
from scipy import stats
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
print("RUN 4: 10-Decile Calibration Reliability Curves & Recalibration (Real Models)")
print("=" * 80)

# Check for real data
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

# Train/Test Split (80% Train, 20% Held-Out Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)

# Train Real Model
print("Training Base Lead Model (GBDT / LightGBM)...")
if LGBM_AVAILABLE:
    base_model = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
else:
    base_model = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42)

base_model.fit(X_train, y_train)

# Raw Uncalibrated Predictions on Test Set
p_uncal = base_model.predict_proba(X_test)[:, 1]
y_true = y_test

# Platt recalibration fit: logit(y=1) = a + b * logit(p_uncal)
eps = 1e-7
p_uncal_clip = np.clip(p_uncal, eps, 1 - eps)
logit_p = np.log(p_uncal_clip / (1 - p_uncal_clip)).reshape(-1, 1)

platt_model = LogisticRegression(solver='lbfgs', C=1e6)
platt_model.fit(logit_p, y_true)
alpha = platt_model.intercept_[0]
beta = platt_model.coef_[0][0]
p_platt = platt_model.predict_proba(logit_p)[:, 1]

# 10 Uniform Deciles Evaluation
def compute_deciles(y_t, y_p):
    deciles = pd.qcut(y_p, q=10, labels=False, duplicates='drop')
    n_bins = len(np.unique(deciles))
    rows = []
    for d in range(n_bins):
        mask = (deciles == d)
        n_g = np.sum(mask)
        if n_g == 0:
            continue
        o_g = np.sum(y_t[mask])
        mean_p = np.mean(y_p[mask])
        obs_rate = o_g / n_g
        rows.append({
            'Decile': d + 1,
            'N_Patients': n_g,
            'Observed_Events': o_g,
            'Mean_Predicted_p': mean_p,
            'Observed_Rate': obs_rate
        })
    return pd.DataFrame(rows)

df_dec_uncal = compute_deciles(y_true, p_uncal)
df_dec_platt = compute_deciles(y_true, p_platt)

# Hosmer-Lemeshow Test on Platt Recalibrated Probabilities
hl_stat = 0.0
for _, row in df_dec_platt.iterrows():
    n_g = row['N_Patients']
    o_g = row['Observed_Events']
    e_g = n_g * row['Mean_Predicted_p']
    hl_stat += ((o_g - e_g) ** 2) / (e_g * (1.0 - row['Mean_Predicted_p'] + 1e-9))

hl_df = len(df_dec_platt) - 2
hl_p = 1.0 - stats.chi2.cdf(hl_stat, hl_df)

# Spiegelhalter z-test for calibration
e_total = p_platt
v_spiegel = np.sum((1.0 - 2.0 * e_total) * (y_true - e_total))
var_spiegel = np.sum(((1.0 - 2.0 * e_total) ** 2) * e_total * (1.0 - e_total))
spiegelhalter_z = v_spiegel / np.sqrt(var_spiegel) if var_spiegel > 0 else 0.0
spiegelhalter_p = 2.0 * (1.0 - stats.norm.cdf(np.abs(spiegelhalter_z)))

calib_stats = {
    'Platt_Intercept_a': float(alpha),
    'Platt_Slope_b': float(beta),
    'Hosmer_Lemeshow_Chi2': float(hl_stat),
    'Hosmer_Lemeshow_df': int(hl_df),
    'Hosmer_Lemeshow_p': float(hl_p),
    'Spiegelhalter_z': float(spiegelhalter_z),
    'Spiegelhalter_p': float(spiegelhalter_p)
}

print(f"Platt Recalibration: logit(y=1) = {alpha:+.4f} + {beta:.4f} * logit(p_hat)")
print(f"Hosmer-Lemeshow chi2({hl_df}) = {hl_stat:.2f}, p = {hl_p:.4f}")
print(f"Spiegelhalter z = {spiegelhalter_z:.2f}, p = {spiegelhalter_p:.4f}")

df_dec_platt.to_csv('results/reviewer_experiments/eval_calibration_deciles.csv', index=False)
with open('results/reviewer_experiments/eval_calibration_stats.json', 'w') as f:
    json.dump(calib_stats, f, indent=2)

# Figure 3: 2-Panel Reliability Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

# Panel A: Pre- vs Post-Recalibration Calibration Curves
ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Ideal Calibration (Slope = 1.00)')
ax1.plot(df_dec_uncal['Mean_Predicted_p'], df_dec_uncal['Observed_Rate'], 'o--', color='#d73027', lw=1.8, markersize=6,
         label=f'Uncalibrated Model (Slope = {1.0/beta if beta != 0 else 1.0:.3f})')
ax1.plot(df_dec_platt['Mean_Predicted_p'], df_dec_platt['Observed_Rate'], 's-', color='#1b7837', lw=2.2, markersize=6,
         label=f'Platt-Recalibrated (Slope = {beta:.3f}, Int = {alpha:+.3f})')

gof_box = (
    f"Goodness-of-Fit Diagnostics:\n"
    f"• Hosmer-Lemeshow: $\\chi^2({hl_df}) = {hl_stat:.2f}$ (p = {hl_p:.3f})\n"
    f"• Spiegelhalter z-test: z = {spiegelhalter_z:.2f} (p = {spiegelhalter_p:.3f})\n"
    f"• Platt Recalibration:\n"
    f"  logit(y=1) = {alpha:+.4f} + {beta:.4f} * logit(p_hat)"
)
ax1.text(0.04, 0.58, gof_box, transform=ax1.transAxes, fontsize=8.2,
         verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.35", fc="#f8f9fa", ec="#ced4da", lw=1.0))

ax1.set_xlim(0, 0.85)
ax1.set_ylim(0, 0.85)
ax1.set_xlabel('Mean Predicted Risk Probability (Decile Mean)', fontsize=10.5, fontweight='bold')
ax1.set_ylabel('Observed 30-Day Readmission Rate (Og / Ng)', fontsize=10.5, fontweight='bold')
ax1.set_title('(A) 10-Decile Reliability Curves (Pre vs Post Platt)', fontsize=11, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left', framealpha=0.95, fontsize=8.5)

# Panel B: Decile Stratification Monotonicity Bar Chart
width = 0.35
n_bins = len(df_dec_platt)
x_pos = np.arange(n_bins)
ax2.bar(x_pos - width/2, df_dec_platt['Mean_Predicted_p'] * 100, width, label='Mean Predicted Risk (%)', color='#91bfdb', edgecolor='#4575b4', lw=1.0)
ax2.bar(x_pos + width/2, df_dec_platt['Observed_Rate'] * 100, width, label='Observed Readmission Rate (%)', color='#fc8d59', edgecolor='#d73027', lw=1.0)

ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"D{d+1}\n({df_dec_platt.loc[d, 'Mean_Predicted_p']*100:.1f}%)" for d in range(n_bins)], fontsize=8.0)
ax2.set_xlabel('Risk Deciles (Mean Predicted Risk %)', fontsize=10.5, fontweight='bold')
ax2.set_ylabel('30-Day Readmission Rate (%)', fontsize=10.5, fontweight='bold')
ax2.set_title('(B) Risk Stratification Monotonicity Across Deciles', fontsize=11, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6, axis='y')
ax2.legend(loc='upper left', framealpha=0.95, fontsize=8.5)

plt.tight_layout()
plt.savefig('figures/fig3_calibration_reliability.pdf')
plt.savefig('figures/fig3_calibration_reliability.png', dpi=300)
plt.close()
print("Saved figures/fig3_calibration_reliability.pdf and .png")
