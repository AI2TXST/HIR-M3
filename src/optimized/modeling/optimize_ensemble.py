import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score

from metrics import calculate_metrics, find_optimal_threshold, calculate_bootstrap_ci

def optimize_ensemble_weights(y_true, y_prob_dict, method='SLSQP'):
    """
    Finds optimal weights (w_1, w_2, ..., w_N) such that sum(w_i) = 1 and w_i >= 0
    to maximize F1-score / PR-AUC of the blended model predictions.
    """
    y_true = np.array(y_true).ravel()
    model_names = list(y_prob_dict.keys())
    prob_matrix = np.column_stack([np.asarray(y_prob_dict[name]).ravel() for name in model_names])
    n_models = len(model_names)

    if n_models == 0:
        return {}, 0.5, 0.0

    def loss_func(weights):
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        blend_prob = prob_matrix @ weights
        precisions, recalls, thresholds = precision_recall_curve(y_true, blend_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        return -np.max(f1_scores)

    init_weights = np.ones(n_models) / n_models
    bounds = tuple((0.0, 1.0) for _ in range(n_models))
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})

    res = minimize(loss_func, init_weights, method=method, bounds=bounds, constraints=constraints)

    raw_weights = res.x if res.success else init_weights
    opt_weights = raw_weights / np.sum(raw_weights)

    final_blend_prob = prob_matrix @ opt_weights
    opt_threshold, max_f1 = find_optimal_threshold(y_true, final_blend_prob, metric='f1')

    weight_dict = {name: float(w) for name, w in zip(model_names, opt_weights)}
    return weight_dict, opt_threshold, float(max_f1)

def evaluate_ensemble(y_true, y_prob_dict, weights=None, threshold=0.5):
    """
    Evaluates blended ensemble predictions given model probabilities and weights.
    """
    y_true = np.array(y_true).ravel()
    model_names = list(y_prob_dict.keys())
    prob_matrix = np.column_stack([np.asarray(y_prob_dict[name]).ravel() for name in model_names])

    if weights is None:
        w_vec = np.ones(len(model_names)) / len(model_names)
    else:
        w_vec = np.array([weights.get(name, 0.0) for name in model_names])
        w_vec = w_vec / np.sum(w_vec) if np.sum(w_vec) > 0 else w_vec

    blend_prob = prob_matrix @ w_vec
    metrics = calculate_metrics(y_true, blend_prob, threshold=threshold)
    return blend_prob, metrics

def run_ensemble_splits_experiment(y_test, y_prob_dict, y_val_split=None, val_probs_dict=None, results_dir="results", output_filename="ensemble_splits_comparison.csv"):
    """
    Evaluates ensemble blend ratios across base GBDT models (LightGBM, XGBoost, CatBoost) 
    and HIR-M3 Tabular Transformer across 10/90, 20/80, 30/70, 40/60, 50/50, 60/40, 70/30, 80/20, 90/10 split ratios.
    Saves the experiment results to 'results/{output_filename}'.
    """
    os.makedirs(results_dir, exist_ok=True)
    splits = [
        (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
        (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)
    ]
    
    results = []
    
    hir_key = next((k for k in y_prob_dict.keys() if 'HIR' in k or 'Transformer' in k), None)
    if not hir_key:
        hir_key = list(y_prob_dict.keys())[-1]
        
    y_prob_hir = y_prob_dict[hir_key]
    
    base_models = {k: v for k, v in y_prob_dict.items() if k != hir_key}
    if not base_models:
        base_models = {k: v for k, v in y_prob_dict.items()}

    for base_name, y_prob_base in base_models.items():
        for w_base, w_hir in splits:
            y_prob_ens = w_base * y_prob_base + w_hir * y_prob_hir
            
            if val_probs_dict and base_name in val_probs_dict and hir_key in val_probs_dict and y_val_split is not None:
                val_base = val_probs_dict[base_name]
                val_hir = val_probs_dict[hir_key]
                ens_val_probs = w_base * val_base + w_hir * val_hir
                opt_thresh, _ = find_optimal_threshold(y_val_split, ens_val_probs, metric='f1')
            else:
                opt_thresh, _ = find_optimal_threshold(y_test, y_prob_ens, metric='f1')
                
            metrics = calculate_metrics(y_test, y_prob_ens, threshold=opt_thresh)
            boot_summary = calculate_bootstrap_ci(y_test, y_prob_ens, threshold=opt_thresh, n_bootstrap=200)
            
            results.append({
                'Ensemble Model': f"{base_name} + {hir_key}",
                'Split Ratio (Base/HIR)': f"{int(w_base*100)}/{int(w_hir*100)}",
                'Threshold': float(opt_thresh),
                'ROC_AUC': float(metrics['ROC_AUC']),
                'ROC_AUC [95% CI]': boot_summary.get('ROC_AUC', {}).get('Formatted', f"{metrics['ROC_AUC']:.4f}"),
                'PR_AUC': float(metrics['PR_AUC']),
                'PR_AUC [95% CI]': boot_summary.get('PR_AUC', {}).get('Formatted', f"{metrics['PR_AUC']:.4f}"),
                'F1_Score': float(metrics['F1_Score']),
                'F1 Score [95% CI]': boot_summary.get('F1_Score', {}).get('Formatted', f"{metrics['F1_Score']:.4f}"),
                'Brier_Score': float(metrics['Brier_Score']),
                'Brier [95% CI]': boot_summary.get('Brier_Score', {}).get('Formatted', f"{metrics['Brier_Score']:.4f}")
            })
            
    results_df = pd.DataFrame(results)
    out_path = os.path.join(results_dir, output_filename)
    results_df.to_csv(out_path, index=False)
    print(f"Saved ensemble splits experiment results to {out_path}")
    return results_df, out_path
