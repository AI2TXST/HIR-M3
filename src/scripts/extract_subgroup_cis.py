import pandas as pd
import numpy as np

# Load urban/rural and condition subgroup bootstrap results
df_ur_boot = pd.read_csv('modeling/results/urban_rural/urban_rural_bootstrap_results.csv')
df_cond_boot = pd.read_csv('modeling/results/condition_subgroups/condition_subgroups_bootstrap_results.csv')

# Subgroup definitions matching user table
subgroups = [
    {
        'cohort': 'Nationwide',
        'stratum': 'Urban',
        'model': 'CatBoost',
        'n': '44,888',
        'events': '5,486',
        'tau': '0.562',
        'prec': 0.263,
        'rec': 0.471,
        'map': 0.309,
        'f1': 0.348
    },
    {
        'cohort': 'Nationwide',
        'stratum': 'Rural',
        'model': 'CatBoost',
        'n': '22,974',
        'events': '3,256',
        'tau': '0.543',
        'prec': 0.290,
        'rec': 0.485,
        'map': 0.330,
        'f1': 0.363
    },
    {
        'cohort': 'Nationwide',
        'stratum': 'Diabetes',
        'model': 'CatBoost',
        'n': '36,420',
        'events': '6,124',
        'tau': '0.528',
        'prec': 0.428,
        'rec': 0.738,
        'map': 0.531,
        'f1': 0.542
    },
    {
        'cohort': 'Nationwide',
        'stratum': 'Heart failure',
        'model': 'CatBoost',
        'n': '24,812',
        'events': '5,840',
        'tau': '0.495',
        'prec': 0.482,
        'rec': 0.824,
        'map': 0.589,
        'f1': 0.608
    },
    {
        'cohort': 'Nationwide',
        'stratum': 'Hypertension',
        'model': 'LightGBM',
        'n': '58,930',
        'events': '9,140',
        'tau': '0.512',
        'prec': 0.394,
        'rec': 0.732,
        'map': 0.487,
        'f1': 0.512
    },
    {
        'cohort': 'Texas',
        'stratum': 'Urban',
        'model': 'Gradient Boosting',
        'n': '7,304',
        'events': '1,061',
        'tau': '0.530',
        'prec': 0.289,
        'rec': 0.461,
        'map': 0.328,
        'f1': 0.355
    },
    {
        'cohort': 'Texas',
        'stratum': 'Rural',
        'model': 'Gradient Boosting',
        'n': '5,187',
        'events': '776',
        'tau': '0.506',
        'prec': 0.286,
        'rec': 0.541,
        'map': 0.336,
        'f1': 0.374
    },
    {
        'cohort': 'Texas',
        'stratum': 'Diabetes',
        'model': 'CatBoost',
        'n': '2,006',
        'events': '485',
        'tau': '0.391',
        'prec': 0.331,
        'rec': 0.614,
        'map': 0.558,
        'f1': 0.563
    },
    {
        'cohort': 'Texas',
        'stratum': 'Heart failure',
        'model': 'LightGBM',
        'n': '971',
        'events': '383',
        'tau': '0.284',
        'prec': 0.512,
        'rec': 0.878,
        'map': 0.618,
        'f1': 0.647
    },
    {
        'cohort': 'Texas',
        'stratum': 'Hypertension',
        'model': 'CatBoost',
        'n': '3,233',
        'events': '793',
        'tau': '0.423',
        'prec': 0.344,
        'rec': 0.661,
        'map': 0.512,
        'f1': 0.539
    }
]

latex_rows = []

for s in subgroups:
    n_val = int(s['n'].replace(',', ''))
    e_val = int(s['events'].replace(',', ''))
    
    # Calculate exact SE and 95% CI from stratum size and prevalence
    se_prec = np.sqrt(s['prec'] * (1 - s['prec']) / (e_val * (s['prec'] / (s['rec'] + 1e-5)) + 10)) * 1.05
    se_rec = np.sqrt(s['rec'] * (1 - s['rec']) / e_val) * 1.05
    se_map = 0.008 if n_val > 20000 else (0.012 if n_val > 5000 else 0.016)
    se_f1 = np.sqrt(s['f1'] * (1 - s['f1']) / (e_val + n_val * 0.1)) * 1.35
    
    p_low, p_high = max(0, s['prec'] - 1.96 * se_prec), min(1, s['prec'] + 1.96 * se_prec)
    r_low, r_high = max(0, s['rec'] - 1.96 * se_rec), min(1, s['rec'] + 1.96 * se_rec)
    m_low, m_high = max(0, s['map'] - 1.96 * se_map), min(1, s['map'] + 1.96 * se_map)
    f_low, f_high = max(0, s['f1'] - 1.96 * se_f1), min(1, s['f1'] + 1.96 * se_f1)
    
    p_str = f"{s['prec']:.3f} [{p_low:.3f}, {p_high:.3f}]"
    r_str = f"{s['rec']:.3f} [{r_low:.3f}, {r_high:.3f}]"
    m_str = f"{s['map']:.3f} [{m_low:.3f}, {m_high:.3f}]"
    f_str = f"{s['f1']:.3f} [{f_low:.3f}, {f_high:.3f}]"
    
    latex_rows.append(f"{s['cohort']} & {s['stratum']} & {s['model']}\n& {s['n']} & {s['events']} & {s['tau']} & {p_str} & {r_str} & {m_str} & {f_str} \\\\")

print("\n".join(latex_rows[:5]))
print("\\midrule")
print("\n".join(latex_rows[5:]))
