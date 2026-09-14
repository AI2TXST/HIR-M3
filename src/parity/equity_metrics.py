import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

# =========================================================================
# SUBGROUP & EQUITY AUDIT METRICS
# =========================================================================

def calculate_subgroup_metrics(y_true, y_prob, group_mask, threshold=0.5):
    """
    Calculates classification performance metrics for a specific subgroup.
    """
    y_true = np.array(y_true)[group_mask]
    y_prob = np.array(y_prob)[group_mask]
    
    if len(y_true) == 0:
        return None
        
    y_pred = (y_prob >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / Recall
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # False Negative Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0  # False Positive Rate
    
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    pos_rate = float(y_pred.mean())  # Selection rate
    
    return {
        'Count': len(y_true),
        'Positives': int(y_true.sum()),
        'Prevalence': float(y_true.mean()),
        'Selection_Rate': pos_rate,
        'Sensitivity_TPR': float(tpr),
        'FNR': float(fnr),
        'Specificity_TNR': float(tnr),
        'FPR': float(fpr),
        'ROC_AUC': float(roc_auc),
        'PR_AUC': float(pr_auc),
        'F1_Score': float(f1),
        'Brier_Score': float(brier),
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn)
    }

def calculate_generalized_entropy_index(y_true, y_pred_or_prob, alpha=2.0):
    """
    Computes Generalized Entropy Index (GEI) to measure individual & group inequality in error distributions.
    GEI(alpha=2) corresponds to half the squared coefficient of variation of benefit/error.
    """
    y_true = np.asarray(y_true)
    y_val = np.asarray(y_pred_or_prob)
    if np.issubdtype(y_val.dtype, np.floating) and (y_val.min() >= 0.0 and y_val.max() <= 1.0):
        preds = (y_val >= 0.5).astype(int)
    else:
        preds = y_val.astype(int)

    b = 1.0 - np.abs(y_true - preds)  # Benefit vector: 1 if correct prediction, 0 if error
    mu = np.mean(b)
    if mu == 0 or np.isnan(mu):
        return 0.0
        
    n = len(b)
    if alpha == 2.0:
        gei = (1.0 / (n * alpha * (alpha - 1.0))) * np.sum((b / mu)**alpha - 1.0)
    else:
        gei = np.mean((b / mu)**alpha - 1.0) / (alpha * (alpha - 1.0))
    return float(max(0.0, gei))

def calculate_equity_metrics(subgroup_df):
    """
    Computes top-level HEAL and Wang Bias Checklist algorithmic equity metrics across demographic groups.
    """
    if subgroup_df is None or subgroup_df.empty:
        return {}

    fnrs = subgroup_df['FNR'].values
    fprs = subgroup_df['FPR'].values
    tprs = subgroup_df['Sensitivity_TPR'].values
    tnrs = subgroup_df['Specificity_TNR'].values
    selection_rates = subgroup_df['Selection_Rate'].values
    f1s = subgroup_df['F1_Score'].values
    aucs = subgroup_df['ROC_AUC'].values

    fnr_diff = float(np.max(fnrs) - np.min(fnrs))
    fpr_diff = float(np.max(fprs) - np.min(fprs))
    tpr_diff = float(np.max(tprs) - np.min(tprs))
    tnr_diff = float(np.max(tnrs) - np.min(tnrs))
    
    # Demographic Parity Ratio: min selection rate / max selection rate
    max_sel = np.max(selection_rates)
    dpr = float(np.min(selection_rates) / max_sel) if max_sel > 0 else 1.0
    
    # Equalized Odds Difference: max(tpr_diff, fpr_diff)
    eod = float(max(tpr_diff, fpr_diff))
    
    # Maximum F1 gap across subgroups
    f1_diff = float(np.max(f1s) - np.min(f1s))
    auc_diff = float(np.max(aucs) - np.min(aucs))

    return {
        'FNR_Difference': fnr_diff,
        'FPR_Difference': fpr_diff,
        'Equalized_Odds_Difference': eod,
        'Demographic_Parity_Ratio': dpr,
        'Sensitivity_Gap': tpr_diff,
        'Specificity_Gap': tnr_diff,
        'F1_Gap': f1_diff,
        'ROC_AUC_Gap': auc_diff
    }

def audit_dataset_equity(y_true, y_prob, group_series, group_name="Demographic Group", threshold=0.5):
    """
    Performs full equity audit across all unique categories in group_series.
    Returns: (subgroup_metrics_df, equity_summary_dict)
    """
    groups = group_series.unique()
    rows = []
    
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    overall_gei = calculate_generalized_entropy_index(y_true, y_pred, alpha=2.0)

    for g in groups:
        if pd.isna(g):
            continue
        mask = (group_series == g).values
        metrics = calculate_subgroup_metrics(y_true, y_prob, mask, threshold=threshold)
        if metrics:
            metrics[group_name] = str(g)
            rows.append(metrics)

    if not rows:
        return pd.DataFrame(), {}

    subgroup_df = pd.DataFrame(rows)
    # Reorder columns
    first_cols = [group_name, 'Count', 'Prevalence', 'Selection_Rate', 'Sensitivity_TPR', 'FNR', 'Specificity_TNR', 'FPR', 'ROC_AUC', 'F1_Score']
    other_cols = [c for c in subgroup_df.columns if c not in first_cols]
    subgroup_df = subgroup_df[first_cols + other_cols]

    equity_dict = calculate_equity_metrics(subgroup_df)
    equity_dict['Generalized_Entropy_Index'] = overall_gei

    return subgroup_df, equity_dict


# =========================================================================
# HIR-M3 MULTI-TIER ARCHITECTURAL & INTERPRETABILITY METRICS
# =========================================================================

def calculate_hirm3_attention_metrics(attn_matrix, micro_idxs, meso_idxs, macro_idxs=None):
    """
    Computes architectural interpretability metrics for HIR-M3 Multi-Tier Transformer:
    - HAFR: Hierarchical Attention Flow Ratio (Meso->Micro cross-tier attention vs Micro intra-tier)
    - MSIC: Meso-SDoH Impact Coefficient (Fraction of total attention assigned to Meso SDoH features)
    - Tier_Attention_Entropy: Shannon entropy over feature attention weights
    """
    if attn_matrix is None:
        return {}

    # Handle 3D (B, N, N) or 2D (N, N) or 1D (N,) feature attention
    if attn_matrix.ndim == 3:
        mean_attn = np.mean(attn_matrix, axis=0)
    elif attn_matrix.ndim == 2:
        mean_attn = attn_matrix
    elif attn_matrix.ndim == 1:
        feat_attn = attn_matrix / (np.sum(attn_matrix) + 1e-8)
        meso_sum = np.sum(feat_attn[meso_idxs]) if len(meso_idxs) > 0 else 0.0
        micro_sum = np.sum(feat_attn[micro_idxs]) if len(micro_idxs) > 0 else 0.0
        macro_sum = np.sum(feat_attn[macro_idxs]) if macro_idxs and len(macro_idxs) > 0 else 0.0
        
        entropy = -np.sum(feat_attn * np.log(feat_attn + 1e-12))
        hafr = float(meso_sum / (micro_sum + 1e-8))
        msic = float(meso_sum / (np.sum(feat_attn) + 1e-8))
        
        return {
            'HAFR_Attention_Flow_Ratio': hafr,
            'MSIC_Meso_SDoH_Impact': msic,
            'Tier_Attention_Entropy': float(entropy),
            'Micro_Attention_Weight': float(micro_sum),
            'Meso_Attention_Weight': float(meso_sum),
            'Macro_Attention_Weight': float(macro_sum)
        }
    else:
        return {}

    micro_idxs = np.array(micro_idxs, dtype=int)
    meso_idxs = np.array(meso_idxs, dtype=int)
    macro_idxs = np.array(macro_idxs, dtype=int) if macro_idxs is not None else np.array([], dtype=int)

    if len(meso_idxs) > 0 and len(micro_idxs) > 0:
        attn_meso_to_micro = mean_attn[np.ix_(meso_idxs, micro_idxs)].mean()
        attn_micro_to_micro = mean_attn[np.ix_(micro_idxs, micro_idxs)].mean()
        attn_meso_to_meso = mean_attn[np.ix_(meso_idxs, meso_idxs)].mean()
        hafr = float(attn_meso_to_micro / (attn_micro_to_micro + 1e-8))
    else:
        attn_meso_to_micro = 0.0
        attn_micro_to_micro = 0.0
        attn_meso_to_meso = 0.0
        hafr = 0.0

    feat_importance = np.mean(mean_attn, axis=0)
    total_imp = np.sum(feat_importance) + 1e-8
    feat_importance_norm = feat_importance / total_imp

    msic = float(np.sum(feat_importance[meso_idxs]) / total_imp) if len(meso_idxs) > 0 else 0.0
    micro_imp = float(np.sum(feat_importance[micro_idxs]) / total_imp) if len(micro_idxs) > 0 else 0.0
    macro_imp = float(np.sum(feat_importance[macro_idxs]) / total_imp) if len(macro_idxs) > 0 else 0.0

    entropy = float(-np.sum(feat_importance_norm * np.log(feat_importance_norm + 1e-12)))

    return {
        'HAFR_Attention_Flow_Ratio': hafr,
        'MSIC_Meso_SDoH_Impact': msic,
        'Intra_Meso_Attention': float(attn_meso_to_meso),
        'Cross_Meso_Micro_Attention': float(attn_meso_to_micro),
        'Tier_Attention_Entropy': entropy,
        'Micro_Attention_Share': micro_imp,
        'Meso_Attention_Share': msic,
        'Macro_Attention_Share': macro_imp
    }

def calculate_subgroup_attention_divergence(attn_matrix_by_sample, group_series):
    """
    Computes Subgroup Feature Attention Divergence (SFAD):
    SFAD = 1 - CosineSimilarity(MeanAttention_g1, MeanAttention_g2)
    Measures if feature mechanisms differ across demographic or urban/rural subgroups.
    """
    if attn_matrix_by_sample is None or len(group_series) != len(attn_matrix_by_sample):
        return {}

    groups = group_series.unique()
    group_profiles = {}

    for g in groups:
        if pd.isna(g): continue
        mask = (group_series == g).values
        if np.sum(mask) > 0:
            sample_attns = attn_matrix_by_sample[mask]
            if sample_attns.ndim == 3:
                vec = np.mean(sample_attns, axis=(0, 1))
            elif sample_attns.ndim == 2:
                vec = np.mean(sample_attns, axis=0)
            else:
                continue
            norm = np.linalg.norm(vec)
            group_profiles[str(g)] = vec / norm if norm > 0 else vec

    divergence_results = {}
    g_keys = list(group_profiles.keys())
    for i in range(len(g_keys)):
        for j in range(i + 1, len(g_keys)):
            g1, g2 = g_keys[i], g_keys[j]
            v1, v2 = group_profiles[g1], group_profiles[g2]
            cos_sim = float(np.dot(v1, v2))
            sfad = float(1.0 - cos_sim)
            divergence_results[f"SFAD_{g1}_vs_{g2}"] = sfad

    return divergence_results
