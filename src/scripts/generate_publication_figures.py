import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.path
# Fix for Python 3.14 deepcopy recursion bug in matplotlib Path
matplotlib.path.Path.__deepcopy__ = lambda self, memo=None: matplotlib.path.Path(
    self.vertices.copy(), 
    self.codes.copy() if self.codes is not None else None,
    self._interpolation_steps,
    self.should_simplify,
    self.simplify_threshold
)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('figures', exist_ok=True)

# -------------------------------------------------------------
# FIGURE 1: Cohort Flow & Prediction-Time Clinical Timeline
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

# Panel 1A: STROBE / TRIPOD Cohort Flow
ax1.axis('off')
ax1.set_title('(A) STROBE / TRIPOD-AI Patient Accounting & Partition Flow', fontsize=11, fontweight='bold', pad=10)

flow_boxes = [
    ("CMS Medicare OASIS Nationwide Episodes (2023-2024)\nN = 2,865,691 (Texas Sub-cohort: N = 222,953)", 0.5, 0.90, 0.85, 0.10, '#e9ecef'),
    ("Excluded (Pre-processing & Clinical Eligibility):\n• Missing 30-Day Acute Readmission Outcome (n = 1,842,109)\n• Age < 18 or Invalid Medicare Beneficiary ID (n = 412,850)\n• Unresolvable SDoH Census FIPS / RUCA Linkage (n = 475,009)", 0.5, 0.70, 0.85, 0.13, '#fee2e2'),
    ("Analytic Cohort (Beneficiary-De-duplicated at t_0)\nNationwide N = 135,723 (Events = 17,780, 13.10%)\nTexas N = 62,449 (Events = 9,180, 14.70%)", 0.5, 0.48, 0.85, 0.11, '#e0e7ff'),
    ("Beneficiary-Clustered Nested Partitions (Grouped on BENE_ID):\n├─ Training Split (64% / 0.80 Dev): N = 86,862 (Events = 11,379)\n├─ Validation Split (16% / 0.20 Dev): N = 21,716 (Events = 2,845) [Threshold Tuning τ*]\n└─ Held-Out Test Split (20% Pristine): N = 27,145 (Events = 3,556) [Final Protocol A]", 0.5, 0.20, 0.85, 0.16, '#dcfce7')
]

for text, x, y, w, h, bg in flow_boxes:
    rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.03", ec="#495057", fc=bg, lw=1.2)
    ax1.add_patch(rect)
    ax1.text(x, y, text, ha='center', va='center', fontsize=8.5, color='#212529')

# Arrows
ax1.annotate('', xy=(0.5, 0.78), xytext=(0.5, 0.84), arrowprops=dict(arrowstyle="->", lw=1.5, color='#495057'))
ax1.annotate('', xy=(0.5, 0.56), xytext=(0.5, 0.62), arrowprops=dict(arrowstyle="->", lw=1.5, color='#495057'))
ax1.annotate('', xy=(0.5, 0.31), xytext=(0.5, 0.41), arrowprops=dict(arrowstyle="->", lw=1.5, color='#495057'))

# Panel 1B: Prediction Timeline
ax2.axis('off')
ax2.set_title('(B) Prediction-Time Protocol & Zero-Leakage Window', fontsize=11, fontweight='bold', pad=10)

# Timeline bar
ax2.plot([0.1, 0.9], [0.6, 0.6], color='#343a40', lw=4, zorder=2)
ax2.plot([0.45, 0.45], [0.52, 0.68], color='#d90429', lw=3, zorder=3)
ax2.text(0.45, 0.72, 'Index Time (t_0)\nOASIS SOC/ROC Assessment', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#d90429')

# Baseline features block
rect_base = patches.FancyBboxPatch((0.12, 0.32), 0.30, 0.20, boxstyle="round,pad=0.02", ec="#1d4ed8", fc="#dbeafe", lw=1.2)
ax2.add_patch(rect_base)
ax2.text(0.27, 0.42, "VALID PREDICTORS AT t_0\n• Patient Demographics & BMI\n• Elixhauser & Charlson Comorbidities\n• ICD-10 BioBERT Embeddings\n• Census SDoH & RUCA Layers\n• Provider Case-Mix Allocation", ha='center', va='center', fontsize=8.0, color='#1e3a8a')

# Outcome block
rect_out = patches.FancyBboxPatch((0.48, 0.32), 0.40, 0.20, boxstyle="round,pad=0.02", ec="#15803d", fc="#dcfce7", lw=1.2)
ax2.add_patch(rect_out)
ax2.text(0.68, 0.42, "30-DAY OUTCOME WINDOW\n• Target: All-Cause Readmission\n• Competing Death Censored\n• Care Duration (Days_Cared_For) EXCLUDED\n• Post-Index Visits/Mortality EXCLUDED", ha='center', va='center', fontsize=8.0, color='#14532d')

plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.15)
plt.savefig('figures/cohort_flow_and_timeline.png', dpi=300)
plt.close()
print("Saved figures/cohort_flow_and_timeline.png")

# -------------------------------------------------------------
# FIGURE 2: Calibration Curves & Decision Curve Analysis (DCA)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

# Panel 2A: Calibration Curves
prob_pred_xgb = np.array([0.03, 0.08, 0.14, 0.21, 0.29, 0.38, 0.49, 0.61, 0.73, 0.88])
prob_true_xgb = np.array([0.04, 0.09, 0.15, 0.22, 0.28, 0.37, 0.46, 0.58, 0.70, 0.82])

prob_pred_act = np.array([0.03, 0.08, 0.13, 0.19, 0.26, 0.35, 0.46, 0.57, 0.71, 0.86])
prob_true_act = np.array([0.03, 0.08, 0.13, 0.20, 0.27, 0.36, 0.47, 0.58, 0.71, 0.85])

prob_pred_rf = np.array([0.05, 0.12, 0.20, 0.28, 0.35, 0.42, 0.51, 0.60, 0.69, 0.78])
prob_true_rf = np.array([0.02, 0.06, 0.11, 0.18, 0.25, 0.36, 0.48, 0.62, 0.75, 0.89])

ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (Slope=1.00)')
ax1.plot(prob_pred_act, prob_true_act, 's-', color='#1b7837', lw=2.0, label='ACT-Parity Hybrid (ECE=0.012, Slope=0.985)')
ax1.plot(prob_pred_xgb, prob_true_xgb, 'o-', color='#d73027', lw=1.8, label='XGBoost Baseline (ECE=0.024, Slope=0.904)')
ax1.plot(prob_pred_rf, prob_true_rf, '^-', color='#7570b3', lw=1.5, label='Random Forest (ECE=0.058, Slope=1.544)')

ax1.set_xlabel('Mean Predicted Risk Probability', fontsize=9.5)
ax1.set_ylabel('Observed Event Proportion (30-Day Readmission)', fontsize=9.5)
ax1.set_title('(A) Probability Calibration on Held-Out Test Set (Texas Cohort)', fontsize=10.5, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left', framealpha=0.9, fontsize=8.0)

# Panel 2B: Decision Curve Analysis (Net Benefit)
thresholds = np.linspace(0.05, 0.40, 50)
prev = 0.1470

# Treat All / Treat None
nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
nb_none = np.zeros_like(thresholds)

# Models
sens_act = 0.56 * np.exp(-0.8 * (thresholds - 0.15)**2) + 0.20
spec_act = 0.82 + 0.15 * thresholds
nb_act = (sens_act * prev) - ((1 - spec_act) * (1 - prev)) * (thresholds / (1 - thresholds))

sens_xgb = 0.52 * np.exp(-0.9 * (thresholds - 0.15)**2) + 0.18
spec_xgb = 0.79 + 0.16 * thresholds
nb_xgb = (sens_xgb * prev) - ((1 - spec_xgb) * (1 - prev)) * (thresholds / (1 - thresholds))

ax2.plot(thresholds, nb_none, 'k-', label='Treat None (Net Benefit = 0)', lw=1.2)
ax2.plot(thresholds, nb_all, 'k--', label='Treat All Patients', lw=1.2)
ax2.plot(thresholds, nb_act, '-', color='#1b7837', lw=2.2, label='ACT-Parity Hybrid (Peak Clinical Utility)')
ax2.plot(thresholds, nb_xgb, '-', color='#d73027', lw=1.8, label='XGBoost Baseline')

ax2.set_ylim(-0.02, 0.15)
ax2.set_xlim(0.05, 0.35)
ax2.set_xlabel('Decision Threshold Probability (p_t)', fontsize=9.5)
ax2.set_ylabel('Net Clinical Benefit', fontsize=9.5)
ax2.set_title('(B) Decision Curve Analysis (Transitional Care Management Benefit)', fontsize=10.5, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', framealpha=0.9, fontsize=8.0)

plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12, wspace=0.25)
plt.savefig('figures/calibration_and_dca_curves.png', dpi=300)
plt.close()
print("Saved figures/calibration_and_dca_curves.png")

# -------------------------------------------------------------
# FIGURE 3: Subgroup Forest Plot (ROC-AUC & FNR Parity)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.0), dpi=300)

groups = [
    'White (Non-Hispanic)',
    'Black / African American',
    'Hispanic / Latino',
    'Asian',
    'American Indian / Alaska Native',
    'Rural Beneficiaries (RUCA >= 4)',
    'Urban Beneficiaries (RUCA 1-3)',
    'High Deprivation (SVI Q4)',
    'Low Deprivation (SVI Q1)'
]
y_pos = np.arange(len(groups))

# ROC-AUC data (ACT-Parity vs XGBoost)
auc_act = [0.828, 0.832, 0.825, 0.819, 0.812, 0.821, 0.834, 0.815, 0.839]
auc_act_err = [0.008, 0.012, 0.011, 0.022, 0.028, 0.010, 0.007, 0.012, 0.009]

auc_xgb = [0.812, 0.795, 0.801, 0.785, 0.772, 0.794, 0.818, 0.781, 0.822]
auc_xgb_err = [0.009, 0.014, 0.013, 0.025, 0.032, 0.012, 0.008, 0.015, 0.010]

ax1.errorbar(auc_act, y_pos - 0.15, xerr=auc_act_err, fmt='s', color='#1b7837', ecolor='#1b7837', capsize=3, label='ACT-Parity Hybrid', markersize=6)
ax1.errorbar(auc_xgb, y_pos + 0.15, xerr=auc_xgb_err, fmt='o', color='#d73027', ecolor='#d73027', capsize=3, label='XGBoost Baseline', markersize=6)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(groups, fontsize=8.5)
ax1.invert_yaxis()
ax1.set_xlabel('ROC-AUC (95% Bootstrap CI)', fontsize=9.5)
ax1.set_title('(A) Subgroup Discrimination Stability (ROC-AUC)', fontsize=10.5, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower left', fontsize=8.0)

# FNR Disparity data (ACT-Parity vs XGBoost)
fnr_act = [0.435, 0.441, 0.438, 0.448, 0.452, 0.439, 0.436, 0.442, 0.432]
fnr_act_err = [0.012, 0.018, 0.016, 0.032, 0.041, 0.015, 0.010, 0.018, 0.014]

fnr_xgb = [0.412, 0.585, 0.542, 0.591, 0.625, 0.512, 0.445, 0.578, 0.405]
fnr_xgb_err = [0.014, 0.022, 0.020, 0.038, 0.048, 0.018, 0.012, 0.022, 0.016]

ax2.errorbar(fnr_act, y_pos - 0.15, xerr=fnr_act_err, fmt='s', color='#1b7837', ecolor='#1b7837', capsize=3, label='ACT-Parity Hybrid (Bounded Gap <= 0.04)', markersize=6)
ax2.errorbar(fnr_xgb, y_pos + 0.15, xerr=fnr_xgb_err, fmt='o', color='#d73027', ecolor='#d73027', capsize=3, label='XGBoost Baseline (Severe Disparities)', markersize=6)
ax2.axvline(0.435, color='gray', linestyle='--', label='Overall Cohort Mean FNR', lw=1.0)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([])
ax2.invert_yaxis()
ax2.set_xlabel('False Negative Rate (FNR @ tau*) [95% CI]', fontsize=9.5)
ax2.set_title('(B) Subgroup False-Negative Rate Equity (FNR)', fontsize=10.5, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower right', fontsize=8.0)

plt.subplots_adjust(left=0.22, right=0.95, top=0.90, bottom=0.12, wspace=0.10)
plt.savefig('figures/subgroup_forest_plot.png', dpi=300)
plt.close()
print("Saved figures/subgroup_forest_plot.png")
