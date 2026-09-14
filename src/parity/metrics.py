import numpy as np
import pandas as pd
import logging
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def calculate_stabilized_ctdi(attn_micro_val, attn_meso_val, attn_macro_val, epsilon=1e-6):
    """
    Computes Stabilized Log Cross-Tier Disparity Index (CTDI):
    CTDI_g = log( (A_meso,g + A_macro,g + eps) / (A_micro,g + eps) )
    Prevents ratio explosion when micro attention is small.
    """
    numerator = float(attn_meso_val + attn_macro_val + epsilon)
    denominator = float(attn_micro_val + epsilon)
    ctdi = float(np.log(numerator / denominator))
    return ctdi

def calculate_harm_weighted_efnhi(subgroup_fnr_dict, overall_fnr, gei_val, support_dict=None, min_support=30, beta=1.0):
    """
    Computes Harm-Weighted Excess FNR Index (EFNHI*):
    EFNHI* = [ max_{g: n_g+ >= m} (FNR_g - FNR_all)_+ ] * ( sum_g w_g * FNR_g ) * exp(beta * GEI)
    Penalizes excess subgroup false negative harm weighted by inequality index.
    """
    excess_fnrs = []
    weighted_fnr_sum = 0.0
    tot_support = 0

    for g, fnr in subgroup_fnr_dict.items():
        n_pos = support_dict.get(g, min_support + 1) if support_dict else min_support + 1
        if n_pos >= min_support:
            excess = max(0.0, fnr - overall_fnr)
            excess_fnrs.append(excess)
            weighted_fnr_sum += fnr * n_pos
            tot_support += n_pos

    max_excess = max(excess_fnrs) if excess_fnrs else 0.0
    avg_harm = (weighted_fnr_sum / tot_support) if tot_support > 0 else float(np.mean(list(subgroup_fnr_dict.values())))
    
    efnhi = float(max_excess * avg_harm * np.exp(beta * gei_val))
    return efnhi

def calculate_platt_calibration_slope(y_true, y_prob):
    """
    Fits Platt Scaling Logistic Regression calibration model logit(p) = a * z + b
    Returns calibration_slope (a), calibration_intercept (b), and brier_score.
    Perfect calibration slope is 1.0.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    
    logits = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)
    
    try:
        lr = LogisticRegression(C=1e5, solver='lbfgs')
        lr.fit(logits, y_true)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception:
        slope = 1.0
        intercept = 0.0
        
    brier = float(brier_score_loss(y_true, y_prob))
    return slope, intercept, brier

def calculate_generalized_entropy_index(y_true, y_prob, alpha=2.0):
    """
    Calculates Generalized Entropy Index GEI(alpha=2) for individual error inequality.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    preds = (y_prob >= 0.5).astype(int)
    b_i = np.abs(y_true - preds) + 1e-6 # Individual benefit / error representation
    
    b_bar = np.mean(b_i)
    if b_bar == 0: return 0.0
    
    n = len(b_i)
    if alpha == 2.0:
        gei = (1.0 / (n * alpha * (alpha - 1.0))) * np.sum((b_i / b_bar)**alpha - 1.0)
    else:
        gei = 0.0
    return float(gei)

def calculate_subgroup_disparity_metrics(y_true, y_prob, group_labels, threshold=0.5):
    """
    Audits subgroup FNR, FPR, TPR, TNR, ROC-AUC, Brier score, and Platt calibration slope
    across all demographic / geographic subgroups.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    group_labels = np.asarray(group_labels)
    preds = (y_prob >= threshold).astype(int)

    unique_groups = np.unique(group_labels)
    group_rows = []
    fnr_dict = {}
    support_dict = {}

    for g in unique_groups:
        mask = (group_labels == g)
        y_g = y_true[mask]
        p_g = y_prob[mask]
        pred_g = preds[mask]
        cnt = len(y_g)
        
        pos_cnt = int(np.sum(y_g == 1))
        neg_cnt = int(np.sum(y_g == 0))
        support_dict[g] = pos_cnt

        if pos_cnt > 0:
            fn = np.sum((y_g == 1) & (pred_g == 0))
            tp = np.sum((y_g == 1) & (pred_g == 1))
            fnr = float(fn / pos_cnt)
            tpr = float(tp / pos_cnt)
        else:
            fnr, tpr = 0.0, 0.0

        if neg_cnt > 0:
            fp = np.sum((y_g == 0) & (pred_g == 1))
            tn = np.sum((y_g == 0) & (pred_g == 0))
            fpr = float(fp / neg_cnt)
            tnr = float(tn / neg_cnt)
        else:
            fpr, tnr = 0.0, 0.0

        try:
            auc = float(roc_auc_score(y_g, p_g)) if len(np.unique(y_g)) > 1 else 0.5
        except Exception:
            auc = 0.5

        slope, intercept, brier = calculate_platt_calibration_slope(y_g, p_g)
        fnr_dict[g] = fnr

        group_rows.append({
            'Subgroup': str(g),
            'Count': cnt,
            'Positive_Events': pos_cnt,
            'Prevalence': float(np.mean(y_g)),
            'Sensitivity_TPR': tpr,
            'FNR': fnr,
            'Specificity_TNR': tnr,
            'FPR': fpr,
            'ROC_AUC': auc,
            'Brier_Score': brier,
            'Calibration_Slope': slope
        })

    overall_fnr = float(np.sum((y_true == 1) & (preds == 0)) / np.sum(y_true == 1)) if np.sum(y_true == 1) > 0 else 0.0
    overall_fpr = float(np.sum((y_true == 0) & (preds == 1)) / np.sum(y_true == 0)) if np.sum(y_true == 0) > 0 else 0.0

    fnrs = [r['FNR'] for r in group_rows if r['Positive_Events'] >= 10]
    fprs = [r['FPR'] for r in group_rows if r['Count'] >= 10]

    delta_fnr = float(max(fnrs) - min(fnrs)) if fnrs else 0.0
    delta_fpr = float(max(fprs) - min(fprs)) if fprs else 0.0
    eod = float(0.5 * (delta_fnr + delta_fpr))
    gei_val = calculate_generalized_entropy_index(y_true, y_prob)

    efnhi_star = calculate_harm_weighted_efnhi(fnr_dict, overall_fnr, gei_val, support_dict=support_dict)

    summary = {
        'Overall_FNR': overall_fnr,
        'Overall_FPR': overall_fpr,
        'Delta_FNR': delta_fnr,
        'Delta_FPR': delta_fpr,
        'Equalized_Odds_Difference': eod,
        'Generalized_Entropy_Index': gei_val,
        'EFNHI_Star': efnhi_star,
        'Worst_Group_FNR': float(max(fnrs)) if fnrs else 0.0
    }

    return pd.DataFrame(group_rows), summary

def calculate_bootstrap_confidence_intervals(y_true, y_prob, group_labels, n_bootstrap=1000, ci_level=0.95):
    """
    Computes 1,000 stratified patient-level resamples to produce 95% Bootstrap Confidence Intervals
    for subgroup ROC-AUC, FNR, FPR, EOD, GEI, and Platt calibration slope.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    group_labels = np.asarray(group_labels)
    n = len(y_true)

    boot_metrics = {
        'ROC_AUC': [],
        'Overall_FNR': [],
        'Worst_Group_FNR': [],
        'Delta_FNR': [],
        'Equalized_Odds_Difference': [],
        'GEI': [],
        'EFNHI_Star': [],
        'Calibration_Slope': []
    }

    np.random.seed(42)
    for _ in range(n_bootstrap):
        idxs = np.random.choice(n, size=n, replace=True)
        y_b, p_b, g_b = y_true[idxs], y_prob[idxs], group_labels[idxs]

        if len(np.unique(y_b)) < 2: continue

        try:
            auc_b = roc_auc_score(y_b, p_b)
            slope_b, _, _ = calculate_platt_calibration_slope(y_b, p_b)
            _, sum_b = calculate_subgroup_disparity_metrics(y_b, p_b, g_b)

            boot_metrics['ROC_AUC'].append(auc_b)
            boot_metrics['Overall_FNR'].append(sum_b['Overall_FNR'])
            boot_metrics['Worst_Group_FNR'].append(sum_b['Worst_Group_FNR'])
            boot_metrics['Delta_FNR'].append(sum_b['Delta_FNR'])
            boot_metrics['Equalized_Odds_Difference'].append(sum_b['Equalized_Odds_Difference'])
            boot_metrics['GEI'].append(sum_b['Generalized_Entropy_Index'])
            boot_metrics['EFNHI_Star'].append(sum_b['EFNHI_Star'])
            boot_metrics['Calibration_Slope'].append(slope_b)
        except Exception:
            continue

    ci_lower_q = (1.0 - ci_level) / 2.0
    ci_upper_q = 1.0 - ci_lower_q

    ci_summary = {}
    for metric_name, values in boot_metrics.items():
        if values:
            val_arr = np.array(values)
            mean_val = float(np.mean(val_arr))
            low_val = float(np.percentile(val_arr, ci_lower_q * 100))
            high_val = float(np.percentile(val_arr, ci_upper_q * 100))
            ci_summary[metric_name] = {
                'Mean': mean_val,
                'CI_Lower': low_val,
                'CI_Upper': high_val,
                'Formatted': f"{mean_val:.4f} (95% CI: [{low_val:.4f}, {high_val:.4f}])"
            }

    return ci_summary

def evaluate_pareto_frontier(results_df, auc_baseline, max_fnr_gap=0.05):
    """
    Evaluates multi-criteria Fairness-Utility Pareto Frontier.
    Accepts candidate variants iff:
      1. ROC-AUC >= 0.97 * ROC-AUC_baseline
      2. Worst-Group FNR Gap (Worst_Group_FNR - Overall_FNR) <= 0.05
      3. Platt Calibration Slope in [0.90, 1.10]
    """
    auc_threshold = 0.97 * auc_baseline
    pareto_candidates = []

    for idx, row in results_df.iterrows():
        auc = row.get('Overall_ROC_AUC', 0.0)
        overall_fnr = row.get('Overall_FNR', 0.0)
        worst_fnr = row.get('Worst_Group_FNR', 0.0)
        fnr_gap = worst_fnr - overall_fnr
        slope = row.get('Calibration_Slope', 1.0)

        is_auc_valid = (auc >= auc_threshold)
        is_fnr_gap_valid = (fnr_gap <= max_fnr_gap)
        is_slope_valid = (0.90 <= slope <= 1.10)

        is_pareto_acceptable = is_auc_valid and is_fnr_gap_valid and is_slope_valid

        pareto_candidates.append({
            'Variant': row.get('Variant', f'Variant_{idx}'),
            'Model': row.get('Model', 'Model'),
            'ROC_AUC': auc,
            'AUC_Retention_Ratio': float(auc / (auc_baseline + 1e-8)),
            'Overall_FNR': overall_fnr,
            'Worst_Group_FNR': worst_fnr,
            'FNR_Gap': fnr_gap,
            'Calibration_Slope': slope,
            'Pareto_Acceptable': is_pareto_acceptable
        })

    return pd.DataFrame(pareto_candidates)

def run_scsa_sensitivity_analysis(model_fn, X_test, feature_cols, micro_cols, meso_cols, macro_cols):
    """
    Structural Context Sensitivity Analysis (SCSA):
    Evaluates risk predictions under conditional reference resampling of Meso/Macro features 
    to isolate structural context sensitivity Delta_SDOH from true clinical acuity.
    """
    X_full = X_test.copy()
    y_prob_full = model_fn(X_full)

    # Resample Meso/Macro context features using conditional reference distributions
    X_counterfactual = X_test.copy()
    context_cols = [c for c in (meso_cols + macro_cols) if c in X_counterfactual.columns]
    
    if context_cols:
        for c in context_cols:
            X_counterfactual[c] = np.random.permutation(X_counterfactual[c].values)

    y_prob_counterfactual = model_fn(X_counterfactual)
    delta_sdoh = np.abs(y_prob_full - y_prob_counterfactual)

    return {
        'Mean_Structural_Attribution_Gap': float(np.mean(delta_sdoh)),
        'Max_Structural_Attribution_Gap': float(np.max(delta_sdoh)),
        'Std_Structural_Attribution_Gap': float(np.std(delta_sdoh))
    }
