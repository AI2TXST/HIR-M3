import os
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    precision_score, recall_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_curve
)

def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """
    Finds the decision threshold that maximizes F1-score or PR-AUC on validation probabilities.
    Robust grid search fallback guarantees a non-zero best threshold when precision_recall_curve is sparse.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    if metric == 'f1':
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        # Fallback grid search if PR curve threshold produces F1 == 0
        if best_f1 == 0.0 or np.isnan(best_f1):
            grid = np.linspace(0.01, 0.99, 99)
            best_grid_f1 = -1.0
            best_grid_thresh = 0.5
            for th in grid:
                preds = (y_prob >= th).astype(int)
                score = f1_score(y_true, preds, zero_division=0)
                if score > best_grid_f1:
                    best_grid_f1 = score
                    best_grid_thresh = th
            if best_grid_f1 > 0.0:
                return float(best_grid_thresh), float(best_grid_f1)

        return float(best_thresh), float(best_f1)
    else:
        return 0.5, 0.0

def calculate_metrics(y_true, y_prob, threshold=0.5):
    """
    Calculates comprehensive classification evaluation metrics.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall

    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)

    return {
        'Threshold': float(threshold),
        'ROC_AUC': float(roc_auc),
        'PR_AUC': float(pr_auc),
        'F1_Score': float(f1),
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(sensitivity),
        'Recall_Sensitivity': float(sensitivity),
        'Specificity': float(specificity),
        'Brier_Score': float(brier),
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }

def calculate_bootstrap_ci(y_true, y_prob, threshold=0.5, n_bootstrap=1000, confidence_level=0.95, seed=42):
    """
    Computes 95% Bootstrap Confidence Intervals (CI) for all classification metrics.
    """
    np.random.seed(seed)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n_samples = len(y_true)

    boot_metrics = {
        'ROC_AUC': [], 'PR_AUC': [], 'F1_Score': [], 'Accuracy': [],
        'Precision': [], 'Recall_Sensitivity': [], 'Specificity': [], 'Brier_Score': []
    }

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_t_boot = y_true[idx]
        y_p_boot = y_prob[idx]

        if len(np.unique(y_t_boot)) < 2:
            continue

        res = calculate_metrics(y_t_boot, y_p_boot, threshold=threshold)
        for key in boot_metrics.keys():
            boot_metrics[key].append(res[key])

    alpha = (1.0 - confidence_level) / 2.0
    summary = {}

    for key, val_list in boot_metrics.items():
        if len(val_list) == 0:
            continue
        lower = np.percentile(val_list, alpha * 100)
        upper = np.percentile(val_list, (1.0 - alpha) * 100)
        mean_val = np.mean(val_list)
        summary[key] = {
            'Mean': float(mean_val),
            'CI_Lower': float(lower),
            'CI_Upper': float(upper),
            'Formatted': f"{mean_val:.4f} ({lower:.4f} - {upper:.4f})"
        }

    return summary

def plot_roc_pr_curves(y_true, y_prob_dict, save_dir="results/figs"):
    """
    Plots side-by-side ROC and Precision-Recall curves for multiple model predictions.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib is not installed. Skipping ROC/PR curve plot generation.")
        return None

    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for model_name, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax1.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})")

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = average_precision_score(y_true, y_prob)
        ax2.plot(rec, prec, label=f"{model_name} (PR-AUC = {pr_auc_val:.3f})")

    ax1.plot([0, 1], [0, 1], 'k--', label='No Skill')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "roc_pr_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path

try:
    from parity.equity_metrics import calculate_subgroup_metrics, calculate_equity_metrics, audit_dataset_equity, calculate_generalized_entropy_index
except ImportError:
    try:
        from equity_metrics import calculate_subgroup_metrics, calculate_equity_metrics, audit_dataset_equity, calculate_generalized_entropy_index
    except ImportError:
        pass
