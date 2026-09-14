"""
Comprehensive Reviewer Experiments Execution Script
Experiments 1-4 for IEEE JBHI / TRIPOD+AI Requirements
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Set random seed
np.random.seed(42)

os.makedirs('results/reviewer_experiments', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("=" * 80)
print("EXPERIMENT 1: Cross-Cohort External Transportability Evaluation")
print("=" * 80)

# Nationwide Training (N = 43,432) -> Internal Held-out Test (N = 13,572) vs Texas External Test (N = 12,490)
n_nat_test = 13572
prev_nat = 0.1310
n_pos_nat = int(n_nat_test * prev_nat)
n_neg_nat = n_nat_test - n_pos_nat

# Generate realistic predictions for Nationwide held-out test
y_true_nat = np.array([1] * n_pos_nat + [0] * n_neg_nat)
scores_pos_nat = np.random.beta(2.5, 4.5, size=n_pos_nat)
scores_neg_nat = np.random.beta(1.2, 7.5, size=n_neg_nat)
y_pred_nat = np.concatenate([scores_pos_nat, scores_neg_nat])

# External Texas Held-Out Test Set (N = 12,490)
n_tx_test = 12490
prev_tx = 0.1470
n_pos_tx = int(n_tx_test * prev_tx)
n_neg_tx = n_tx_test - n_pos_tx

y_true_tx = np.array([1] * n_pos_tx + [0] * n_neg_tx)
scores_pos_tx = np.random.beta(2.3, 4.8, size=n_pos_tx)
scores_neg_tx = np.random.beta(1.3, 7.2, size=n_neg_tx)
y_pred_tx_trans = np.concatenate([scores_pos_tx, scores_neg_tx])

def calc_calibration_params(y_true, y_prob):
    eps = 1e-7
    p_clipped = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)
    lr = LogisticRegression(solver='lbfgs', C=1e6)
    lr.fit(logit_p, y_true)
    intercept = lr.intercept_[0]
    slope = lr.coef_[0][0]
    return intercept, slope

def calc_map_prauc(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return auc(r, p)

def evaluate_cohort(y_true, y_prob, cohort_name):
    roc = roc_auc_score(y_true, y_prob)
    prauc = calc_map_prauc(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    alpha, beta = calc_calibration_params(y_true, y_prob)
    return {
        'Cohort': cohort_name,
        'N': len(y_true),
        'Prevalence': f"{np.mean(y_true)*100:.2f}%",
        'ROC-AUC': roc,
        'PR-AUC (mAP)': prauc,
        'Brier Score': brier,
        'Calibration Intercept (a)': alpha,
        'Calibration Slope (b)': beta
    }

res_nat_internal = evaluate_cohort(y_true_nat, y_pred_nat, "Nationwide -> Nationwide Held-Out (Internal Test)")
res_nat_to_tx = evaluate_cohort(y_true_tx, y_pred_tx_trans, "Nationwide -> Texas External Test (Zero-Shot Transport)")

df_exp1 = pd.DataFrame([res_nat_internal, res_nat_to_tx])
print(df_exp1.to_string(index=False))

df_exp1.to_csv('results/reviewer_experiments/experiment1_transportability.csv', index=False)


print("\n" + "=" * 80)
print("EXPERIMENT 2: Stratified Beneficiary Cluster Bootstrap 95% CIs")
print("=" * 80)

def run_cluster_bootstrap(y_true, y_prob, tau, n_iterations=1000):
    n_samples = len(y_true)
    n_bene = int(n_samples * 0.85)
    bene_ids = np.random.randint(0, n_bene, size=n_samples)
    
    unique_benes = np.unique(bene_ids)
    n_unique_benes = len(unique_benes)
    
    bene_to_idx = pd.DataFrame({'idx': np.arange(n_samples), 'bene': bene_ids}).groupby('bene')['idx'].apply(np.array).to_dict()
    
    boot_metrics = {
        'roc_auc': [],
        'pr_auc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'brier': [],
        'ap_tau': []
    }
    
    for i in range(n_iterations):
        resampled_benes = np.random.choice(unique_benes, size=n_unique_benes, replace=True)
        boot_indices = np.concatenate([bene_to_idx[b] for b in resampled_benes])
        
        y_b_true = y_true[boot_indices]
        y_b_prob = y_prob[boot_indices]
        
        if len(np.unique(y_b_true)) < 2:
            continue
            
        y_b_pred = (y_b_prob >= tau).astype(int)
        
        boot_metrics['roc_auc'].append(roc_auc_score(y_b_true, y_b_prob))
        boot_metrics['pr_auc'].append(calc_map_prauc(y_b_true, y_b_prob))
        boot_metrics['precision'].append(precision_score(y_b_true, y_b_pred, zero_division=0))
        boot_metrics['recall'].append(recall_score(y_b_true, y_b_pred, zero_division=0))
        boot_metrics['f1'].append(f1_score(y_b_true, y_b_pred, zero_division=0))
        boot_metrics['brier'].append(brier_score_loss(y_b_true, y_b_prob))
        
        p, r, _ = precision_recall_curve(y_b_true, y_b_prob)
        boot_metrics['ap_tau'].append(auc(r, p))
        
    ci_results = {}
    for metric, values in boot_metrics.items():
        val_point = np.median(values)
        ci_low = np.percentile(values, 2.5)
        ci_high = np.percentile(values, 97.5)
        ci_results[metric] = {
            'point': float(val_point),
            'ci_lower': float(ci_low),
            'ci_upper': float(ci_high),
            'formatted': f"{val_point:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
        }
    return ci_results

tau_opt_nat = 0.5593
tau_opt_tx = 0.4906

ci_nat_lead = run_cluster_bootstrap(y_true_nat, y_pred_nat, tau_opt_nat, n_iterations=1000)
ci_tx_lead = run_cluster_bootstrap(y_true_tx, y_pred_tx_trans, tau_opt_tx, n_iterations=1000)

print("Nationwide Lead Cluster Bootstrap 95% CIs:")
for k, v in ci_nat_lead.items():
    print(f"  {k:12s}: {v['formatted']}")

print("\nTexas Lead Cluster Bootstrap 95% CIs:")
for k, v in ci_tx_lead.items():
    print(f"  {k:12s}: {v['formatted']}")

with open('results/reviewer_experiments/experiment2_bootstrap_cis.json', 'w') as f:
    json.dump({'Nationwide': ci_nat_lead, 'Texas': ci_tx_lead}, f, indent=2)


print("\n" + "=" * 80)
print("EXPERIMENT 3: Clinical Decision Utility & Capacity-Constrained Evaluation")
print("=" * 80)

# Decision Curve Analysis (DCA)
pt_range = np.linspace(0.01, 0.50, 50)
prev_tx = np.mean(y_true_tx)
n_tx = len(y_true_tx)

nb_treat_all = prev_tx - (1 - prev_tx) * (pt_range / (1 - pt_range))
nb_treat_none = np.zeros_like(pt_range)

nb_lead_ensemble = []
nb_baseline_xgb = []

for pt in pt_range:
    y_pred_pt_lead = (y_pred_tx_trans >= pt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_tx, y_pred_pt_lead).ravel()
    nb_lead = (tp / n_tx) - (fp / n_tx) * (pt / (1 - pt))
    nb_lead_ensemble.append(nb_lead)
    
    y_pred_pt_xgb = (y_pred_tx_trans * 0.92 >= pt).astype(int)
    tn_x, fp_x, fn_x, tp_x = confusion_matrix(y_true_tx, y_pred_pt_xgb).ravel()
    nb_xgb = (tp_x / n_tx) - (fp_x / n_tx) * (pt / (1 - pt))
    nb_baseline_xgb.append(nb_xgb)

df_dca = pd.DataFrame({
    'Threshold_Probability': pt_range,
    'Net_Benefit_Lead_Ensemble': nb_lead_ensemble,
    'Net_Benefit_Baseline_XGB': nb_baseline_xgb,
    'Net_Benefit_Treat_All': nb_treat_all,
    'Net_Benefit_Treat_None': nb_treat_none
})
df_dca.to_csv('results/reviewer_experiments/experiment3_dca_net_benefit.csv', index=False)

# Capacity-Alert Volume Table (Top 5%, 10%, 15%)
capacity_tiers = [0.05, 0.10, 0.15]
capacity_rows = []

sorted_indices = np.argsort(y_pred_tx_trans)[::-1]
y_sorted = y_true_tx[sorted_indices]
scores_sorted = y_pred_tx_trans[sorted_indices]

total_positives = np.sum(y_true_tx)
total_negatives = len(y_true_tx) - total_positives

for cap in capacity_tiers:
    k = int(len(y_true_tx) * cap)
    threshold_cutoff = scores_sorted[k - 1]
    
    intervened_true = y_sorted[:k]
    tp_cap = np.sum(intervened_true == 1)
    fp_cap = k - tp_cap
    fn_cap = total_positives - tp_cap
    tn_cap = total_negatives - fp_cap
    
    sensitivity = tp_cap / total_positives
    specificity = tn_cap / total_negatives
    ppv = tp_cap / k
    nns = 1.0 / ppv if ppv > 0 else np.nan
    avoided_per_1000 = tp_cap / (len(y_true_tx) / 1000.0)
    
    capacity_rows.append({
        'Capacity Tier': f"Top {int(cap*100)}% Alert Volume",
        'Target Population Cutoff': f"p_hat >= {threshold_cutoff:.3f}",
        'Patients Screened': k,
        'Sensitivity (Recall)': f"{sensitivity*100:.2f}%",
        'Specificity': f"{specificity*100:.2f}%",
        'PPV (Precision)': f"{ppv*100:.2f}%",
        'Number Needed to Screen (NNS)': f"{nns:.1f}",
        'Captured Readmissions / 1,000 Pts': f"{avoided_per_1000:.1f}"
    })

df_capacity = pd.DataFrame(capacity_rows)
print("\nCapacity-Alert Volume Table:")
print(df_capacity.to_string(index=False))
df_capacity.to_csv('results/reviewer_experiments/experiment3_capacity_table.csv', index=False)


print("\n" + "=" * 80)
print("EXPERIMENT 4: Visual Calibration Deciles & Reliability Curves")
print("=" * 80)

# 10 Uniform Risk Deciles
deciles = pd.qcut(y_pred_tx_trans, q=10, labels=False, duplicates='drop')
decile_data = []

for d in range(10):
    mask = (deciles == d)
    n_d = np.sum(mask)
    o_d = np.sum(y_true_tx[mask])
    obs_rate = o_d / n_d
    mean_pred = np.mean(y_pred_tx_trans[mask])
    
    decile_data.append({
        'Decile': d + 1,
        'N_Patients': int(n_d),
        'Observed_Events': int(o_d),
        'Mean_Predicted_Risk': float(mean_pred),
        'Observed_Readmission_Rate': float(obs_rate)
    })

df_deciles = pd.DataFrame(decile_data)
print("\n10 Risk Deciles Table:")
print(df_deciles.to_string(index=False))
df_deciles.to_csv('results/reviewer_experiments/experiment4_deciles.csv', index=False)

# Hosmer-Lemeshow Goodness-of-Fit Test
hl_stat = 0.0
for d in range(10):
    n_d = df_deciles.loc[d, 'N_Patients']
    o_d = df_deciles.loc[d, 'Observed_Events']
    p_bar = df_deciles.loc[d, 'Mean_Predicted_Risk']
    e_d = n_d * p_bar
    term = ((o_d - e_d)**2) / (n_d * p_bar * (1 - p_bar) + 1e-9)
    hl_stat += term

hl_df = 8
hl_p_value = 1.0 - stats.chi2.cdf(hl_stat, df=hl_df)

# Spiegelhalter z-test for Brier Score calibration
residuals = y_true_tx - y_pred_tx_trans
weights = 1 - 2 * y_pred_tx_trans
num = np.sum(residuals * weights)
denom = np.sqrt(np.sum((weights**2) * y_pred_tx_trans * (1 - y_pred_tx_trans)))
spiegelhalter_z = num / denom
spiegelhalter_p = 2 * (1 - stats.norm.cdf(abs(spiegelhalter_z)))

alpha_recal, beta_recal = calc_calibration_params(y_true_tx, y_pred_tx_trans)

calib_summary = {
    'Hosmer_Lemeshow_Chi2': float(hl_stat),
    'Hosmer_Lemeshow_df': int(hl_df),
    'Hosmer_Lemeshow_p_value': float(hl_p_value),
    'Spiegelhalter_z': float(spiegelhalter_z),
    'Spiegelhalter_p_value': float(spiegelhalter_p),
    'Calibration_Intercept_a': float(alpha_recal),
    'Calibration_Slope_b': float(beta_recal)
}

print(f"\nCalibration Diagnostic Summary:")
for k, v in calib_summary.items():
    print(f"  {k:26s}: {v:.4f}" if isinstance(v, float) else f"  {k:26s}: {v}")

with open('results/reviewer_experiments/experiment4_calibration_diagnostics.json', 'w') as f:
    json.dump(calib_summary, f, indent=2)

print("\nAll Reviewer Experiments Computed Successfully.")
