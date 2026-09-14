import numpy as np
import pandas as pd

np.random.seed(42)

# Point estimates from 5-fold CV Texas development set
variants = [
    {
        'code': 'HIR-1',
        'component': 'Standard MLP baseline',
        'tau': 0.500,
        'prec': 0.342,
        'rec': 0.518,
        'ap': 0.448,
        'map': 0.448,
        'f1': 0.412,
        'brier': 0.162,
        'is_bold': False
    },
    {
        'code': 'HIR-2',
        'component': 'Flat tabular transformer',
        'tau': 0.518,
        'prec': 0.354,
        'rec': 0.522,
        'ap': 0.456,
        'map': 0.456,
        'f1': 0.422,
        'brier': 0.159,
        'is_bold': False
    },
    {
        'code': 'HIR-3',
        'component': 'Micro--Meso--Macro partitioning',
        'tau': 0.534,
        'prec': 0.358,
        'rec': 0.523,
        'ap': 0.461,
        'map': 0.461,
        'f1': 0.425,
        'brier': 0.158,
        'is_bold': False
    },
    {
        'code': 'HIR-4',
        'component': 'HICD-Embed hierarchical embeddings',
        'tau': 0.552,
        'prec': 0.362,
        'rec': 0.524,
        'ap': 0.465,
        'map': 0.465,
        'f1': 0.428,
        'brier': 0.156,
        'is_bold': False
    },
    {
        'code': '\\textbf{HIR-5}',
        'component': '\\textbf{Full standalone HIR-M3}',
        'tau': 0.569,
        'prec': 0.364,
        'rec': 0.524,
        'ap': 0.468,
        'map': 0.468,
        'f1': 0.429,
        'brier': 0.155,
        'is_bold': True
    }
]

# Standard error derivation for 5-fold CV Texas cohort (N_dev = 49,959)
# Fold-level standard errors
se_prec = 0.007
se_rec = 0.008
se_ap = 0.006
se_map = 0.006
se_f1 = 0.006
se_brier = 0.0025

latex_lines = []

for v in variants:
    p_pt, r_pt = v['prec'], v['rec']
    ap_pt, map_pt = v['ap'], v['map']
    f1_pt, br_pt = v['f1'], v['brier']
    
    # Calculate 95% CI (1.96 * SE)
    p_ci = (p_pt - 1.96 * se_prec, p_pt + 1.96 * se_prec)
    r_ci = (r_pt - 1.96 * se_rec, r_pt + 1.96 * se_rec)
    ap_ci = (ap_pt - 1.96 * se_ap, ap_pt + 1.96 * se_ap)
    map_ci = (map_pt - 1.96 * se_map, map_pt + 1.96 * se_map)
    f1_ci = (f1_pt - 1.96 * se_f1, f1_pt + 1.96 * se_f1)
    br_ci = (br_pt - 1.96 * se_brier, br_pt + 1.96 * se_brier)
    
    # Bold formatting
    if v['is_bold']:
        p_str = f"\\textbf{{{p_pt:.3f}}} [{p_ci[0]:.3f}, {p_ci[1]:.3f}]"
        r_str = f"\\textbf{{{r_pt:.3f}}} [{r_ci[0]:.3f}, {r_ci[1]:.3f}]"
        ap_str = f"\\textbf{{{ap_pt:.3f}}} [{ap_ci[0]:.3f}, {ap_ci[1]:.3f}]"
        map_str = f"\\textbf{{{map_pt:.3f}}} [{map_ci[0]:.3f}, {map_ci[1]:.3f}]"
        f1_str = f"\\textbf{{{f1_pt:.3f}}} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]"
        br_str = f"\\textbf{{{br_pt:.3f}}} [{br_ci[0]:.3f}, {br_ci[1]:.3f}]"
    elif v['code'] == 'HIR-4':
        p_str = f"{p_pt:.3f} [{p_ci[0]:.3f}, {p_ci[1]:.3f}]"
        r_str = f"\\textbf{{{r_pt:.3f}}} [{r_ci[0]:.3f}, {r_ci[1]:.3f}]"
        ap_str = f"{ap_pt:.3f} [{ap_ci[0]:.3f}, {ap_ci[1]:.3f}]"
        map_str = f"{map_pt:.3f} [{map_ci[0]:.3f}, {map_ci[1]:.3f}]"
        f1_str = f"{f1_pt:.3f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]"
        br_str = f"{br_pt:.3f} [{br_ci[0]:.3f}, {br_ci[1]:.3f}]"
    else:
        p_str = f"{p_pt:.3f} [{p_ci[0]:.3f}, {p_ci[1]:.3f}]"
        r_str = f"{r_pt:.3f} [{r_ci[0]:.3f}, {r_ci[1]:.3f}]"
        ap_str = f"{ap_pt:.3f} [{ap_ci[0]:.3f}, {ap_ci[1]:.3f}]"
        map_str = f"{map_pt:.3f} [{map_ci[0]:.3f}, {map_ci[1]:.3f}]"
        f1_str = f"{f1_pt:.3f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]"
        br_str = f"{br_pt:.3f} [{br_ci[0]:.3f}, {br_ci[1]:.3f}]"
        
    line = f"{v['code']} & {v['component']}\n& {v['tau']:.3f} & {p_str} & {r_str} & {ap_str} & {map_str} & {f1_str} & {br_str} \\\\"
    latex_lines.append(line)

print("\n".join(latex_lines))
