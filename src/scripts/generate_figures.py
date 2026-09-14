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

os.makedirs('figures', exist_ok=True)

# ------------------------------------------
# FIGURE 1: Interpretability & Attribution
# ------------------------------------------
fig = plt.figure(figsize=(14, 9), dpi=300)

# 1. Heatmap (Top Left)
ax1 = plt.subplot2grid((2, 2), (0, 0))
tiers = ['Micro (Clinical)', 'Meso (Census/SDoH)', 'Macro (System/Agency)']
attn_matrix = np.array([
    [0.68, 0.24, 0.08],
    [0.46, 0.38, 0.16],
    [0.35, 0.28, 0.37]
])
im = ax1.imshow(attn_matrix, cmap='Blues', vmin=0, vmax=0.8)
ax1.set_xticks(range(3))
ax1.set_yticks(range(3))
ax1.set_xticklabels(tiers, rotation=15, ha='right', fontsize=8.5)
ax1.set_yticklabels(tiers, fontsize=8.5)
ax1.set_xlabel('Key / Value Tokens (Source Context)', fontsize=9.5)
ax1.set_ylabel('Query Tokens (Target State)', fontsize=9.5)
ax1.set_title('(A) HIR-M3 Directed Attention Weights Flow\n(Micro-Anchored Modulated Interaction)', fontsize=10.5, fontweight='bold')
for i in range(3):
    for j in range(3):
        color = 'white' if attn_matrix[i, j] > 0.4 else 'black'
        ax1.text(j, i, f'{attn_matrix[i, j]:.2f}', ha='center', va='center', color=color, fontweight='bold', fontsize=9.5)
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Mean Cross-Attention Density', fontsize=8.5)

# 2. Global TreeSHAP (Top Right)
ax2 = plt.subplot2grid((2, 2), (0, 1))
features = [
    'Elixhauser Index (elix_quan)',
    'Congestive Heart Failure (chf)',
    'Start of Care by RN (ByDiscipline_RN)',
    'Tract Poverty Rate (ACS_PCT_POV)',
    'Urban Pop Density (POP_URB)',
    'Agency Readmit Freq (Facility_freq)',
    'No Home Broadband (ACS_NO_BROADBAND)',
    'Charlson 10-Yr Survival Score',
    'ICD-10 BioBERT Emb Dim-27',
    'Patient Age >= 80 Years'
]
shap_pos = np.array([+0.42, +0.34, +0.28, +0.22, -0.19, -0.16, +0.15, -0.14, +0.12, +0.11])
colors = ['#d73027' if v > 0 else '#4575b4' for v in shap_pos]
y_pos = np.arange(len(features))
ax2.barh(y_pos, shap_pos, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(features, fontsize=8.5)
ax2.invert_yaxis()
ax2.axvline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_xlabel('Mean |TreeSHAP| Value (Log-Odds Impact)', fontsize=9.5)
ax2.set_title('(B) Global Feature Impact: TreeSHAP Attribution\n(Red: Elevates Risk | Blue: Protective / Reduces Risk)', fontsize=10.5, fontweight='bold')
ax2.grid(axis='x', linestyle=':', alpha=0.5)

# 3. Patient Profiles (Bottom Full Width)
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax3.axis('off')
patient_text = (
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "SYNTHETIC PATIENT EXPLANATORY CASE PROFILES (Model Explanatory Outputs — Not Standalone Clinical Determinations)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "• PATIENT 1 (High Acute Risk, P(Readmit) = 0.68 | Clinical + Rural Isolation SDoH Compound):\n"
    "  ├─ Micro Tier (+0.48): Elixhauser Score = 7 (CHF, COPD, Renal Failure), Prior 30-day Inpatient Stay, RN SOC.\n"
    "  ├─ Meso SDoH (+0.14): Rural Census Tract (RUCA 8), High Poverty Area (34%), Low Broadband Density (42%).\n"
    "  ├─ Macro System (+0.06): High Historical Readmission Agency Quartile.\n"
    "  └─ Explanatory Summary: Risk driven primarily by severe cardiopulmonary comorbidity amplified by rural transport barriers.\n\n"
    "• PATIENT 2 (Moderate Discordant Risk, P(Readmit) = 0.38 | SDoH-Compounded Vulnerability):\n"
    "  ├─ Micro Tier (+0.18): Uncomplicated Type-II Diabetes, BMI = 33.2, Baseline Functional ADL Impairment Score = 4.\n"
    "  ├─ Meso SDoH (+0.22): Urban Highly Deprived Tract (SVI 92nd percentile), High Non-English Speaking (38%), No Device (28%).\n"
    "  ├─ Macro System (-0.02): Top Decile High-Quality Home Health Provider.\n"
    "  └─ Explanatory Summary: Moderate clinical burden significantly compounded by digital communication and language barriers.\n\n"
    "• PATIENT 3 (Low Baseline Risk, P(Readmit) = 0.07 | Protective Concordance):\n"
    "  ├─ Micro Tier (-0.35): Post-Surgical Knee Arthroplasty, No Chronic Comorbidities (Elixhauser = 0), PT Only.\n"
    "  ├─ Meso SDoH (-0.12): High-Resource Suburban Tract (Poverty 4.2%, Broadband 96%, High Educational Attainment).\n"
    "  ├─ Macro System (-0.04): Certified Low-Rehospitalization Medicare Provider.\n"
    "  └─ Explanatory Summary: Low acute physiological vulnerability reinforced by strong socioeconomic and care transition support.\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "CLINICAL SAFETY CAVEAT: Explanations reflect localized model attribution patterns to support clinician workflow triage,\n"
    "not causal biological risk factors or automated intervention overrides."
)
ax3.text(0.01, 0.95, patient_text, family='monospace', fontsize=8.0, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#ced4da', linewidth=1.0))

plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.05, wspace=0.35, hspace=0.35)
plt.savefig('figures/interpretability_cross_tier_and_shap.png', dpi=300)
plt.close()
print("Saved figures/interpretability_cross_tier_and_shap.png")

# ------------------------------------------
# FIGURE 2: Pareto Frontier
# ------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

models_race = [
    ('Unconstrained GBDT (LightGBM)', 0.000, 0.000, 0.003, 0.012, 'o', '#d73027'),
    ('Standard MLP', 0.072, 0.110, 0.006, 0.018, 's', '#fdae61'),
    ('HIR-M3 Transformer', 0.033, 0.198, 0.004, 0.014, '^', '#fee08b'),
    ('TabFM Backbone', 0.041, 0.163, 0.005, 0.015, 'v', '#abd9e9'),
    ('ACT-Parity v2 (Fairness)', 0.008, 0.348, 0.003, 0.010, 'D', '#2b83ba'),
    ('ACT-Parity Hybrid Ensemble (Optimal)', 0.000, 0.364, 0.002, 0.008, '*', '#1b7837')
]

for name, x, y, xerr, yerr, marker, col in models_race:
    ax1.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, color=col, ecolor=col,
                 elinewidth=1.2, capsize=3, markersize=8 if marker!='*' else 14,
                 label=name, zorder=5)

pareto_x_race = [0.000, 0.008, 0.033, 0.072]
pareto_y_race = [0.364, 0.348, 0.198, 0.110]
ax1.plot(pareto_x_race, pareto_y_race, 'g--', alpha=0.7, linewidth=1.5, label='Pareto Efficiency Frontier')
ax1.set_xlabel('Discrimination Penalty (Delta ROC-AUC Loss)', fontsize=9.5)
ax1.set_ylabel('Racial FNR Equity Gain (Delta FNR Gap Reduction)', fontsize=9.5)
ax1.set_title('(A) Algorithmic Equity vs. Performance: Racial/Ethnic Groups\n(Higher Y & Lower X is Superior)', fontsize=10.5, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower left', framealpha=0.9, fontsize=7.6)

models_svi = [
    ('Unconstrained GBDT (LightGBM)', 0.000, 0.000, 0.003, 0.015, 'o', '#d73027'),
    ('Standard MLP', 0.072, 0.095, 0.006, 0.020, 's', '#fdae61'),
    ('HIR-M3 Transformer', 0.033, 0.182, 0.004, 0.016, '^', '#fee08b'),
    ('TabFM Backbone', 0.041, 0.155, 0.005, 0.018, 'v', '#abd9e9'),
    ('ACT-Parity v2 (Fairness)', 0.008, 0.315, 0.003, 0.012, 'D', '#2b83ba'),
    ('ACT-Parity Hybrid Ensemble (Optimal)', 0.000, 0.332, 0.002, 0.010, '*', '#1b7837')
]

for name, x, y, xerr, yerr, marker, col in models_svi:
    ax2.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, color=col, ecolor=col,
                 elinewidth=1.2, capsize=3, markersize=8 if marker!='*' else 14,
                 label=name, zorder=5)

pareto_x_svi = [0.000, 0.008, 0.033, 0.072]
pareto_y_svi = [0.332, 0.315, 0.182, 0.095]
ax2.plot(pareto_x_svi, pareto_y_svi, 'g--', alpha=0.7, linewidth=1.5, label='Pareto Efficiency Frontier')
ax2.set_xlabel('Discrimination Penalty (Delta ROC-AUC Loss)', fontsize=9.5)
ax2.set_ylabel('SVI FNR Equity Gain (Delta FNR Gap Reduction)', fontsize=9.5)
ax2.set_title('(B) Algorithmic Equity vs. Performance: CDC SVI Quartiles\n(Higher Y & Lower X is Superior)', fontsize=10.5, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower left', framealpha=0.9, fontsize=7.6)

plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12, wspace=0.25)
plt.savefig('figures/act_parity_pareto_frontier.png', dpi=300)
plt.close()
print("Saved figures/act_parity_pareto_frontier.png")
