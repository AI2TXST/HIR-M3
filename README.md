# HIR-M3: Hierarchy-Aware Tabular Modeling and ACT-Parity Evaluation for 30-Day Acute-Care Utilization After Home Health Care

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![TRIPOD-AI Compliant](https://img.shields.io/badge/TRIPOD--AI-Compliant-brightgreen.svg)](docs/TRIPOD_AI_CHECKLIST_AND_S1_MATRIX.md)

This repository provides the official implementation, experimental pipelines, and evaluation suite for the study:  
**"HIR-M3: Hierarchy-Aware Tabular Modeling and ACT-Parity Evaluation for 30-Day Acute-Care Utilization After Home Health Care"**.

---

## 🔬 Study Architecture

The research framework investigates 30-day post-acute home health hospital readmission prediction using **CMS OASIS national data** ($N \approx 100,000$) and a **Texas statewide cohort** ($N \approx 50,000$) across three integrated tasks:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THREE-TASK RESEARCH DESIGN                                   │
├────────────────────────────────┬───────────────────────────────┬───────────────────────────────┤
│            TASK 1              │            TASK 2             │            TASK 3             │
│ Baseline Model Benchmarking &  │ Weighted Deep Ensembling with │  ACT-Parity Optimization &    │
│ Subgroup Operating Trade-offs  │ Hierarchical Tabular (HIR-M3) │ Comprehensive Equity Auditing │
├────────────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ • 10 Model Architectures       │ • Base-to-HIR-M3 Ratios       │ • Augmented Lagrangian Method │
│ • Nationwide vs. Texas         │ • Incremental Ensemble Value  │ • Multi-Attribute Parity      │
│ • Rural vs. Urban Disparities  │ • Nationwide: 90:10 Blend     │ • FPSA Threshold Robustness   │
│ • 7 Comorbidity Subgroups      │ • Texas: 30:70 Blend          │ • Cross-Cohort Transport      │
└────────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

---

## 📂 Repository Structure

The repository is organized into three clean root directories (`data/`, `docs/`, `src/`):

```
.
├── README.md                                  # Repository overview and reproduction guide
├── LICENSE                                    # MIT License
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Git ignore rules for DUA & repository hygiene
│
├── data/                                      # Data documentation & benchmark metrics
│   ├── README.md                              # CMS OASIS & SDoH data access & replication guide
│   ├── best_hir_m3.pth                        # Pretrained model weights
│   ├── hir_m3_metrics.csv                     # Model performance evaluation metrics
│   └── hir_sdoh_ablation_results.csv          # SDoH tier ablation study results
│
├── docs/                                      # Clinical reports, supplementary files & figures
│   ├── figures/                               # Publication figures (PDF & PNG)
│   ├── TRIPOD_AI_CHECKLIST_AND_S1_MATRIX.md   # TRIPOD-AI checklist & verification matrix
│   ├── COMPLETE_STUDY_RESULTS_REPORT.md       # Comprehensive study results & subgroup reports
│   ├── supplementary_materials.tex            # Supplementary derivations and extended tables
│   ├── tasks.md                               # Complete three-task technical report
│   └── *.md / *.pdf                           # Extended clinical & architectural reports
│
└── src/                                       # Source code, execution pipelines & SLURM scripts
    ├── run_all_slurm_pipeline.sh              # Master SLURM pipeline execution script
    │
    ├── 01_build_clean_datasets.slurm          # Stage 01: Cohort extraction & SDoH linkage
    ├── 02_feature_selection.slurm             # Stage 02: Multi-method feature selection
    ├── 03_primary_modeling.slurm              # Stage 03: GBDT baseline modeling
    ├── 04_neural_modeling.slurm               # Stage 04: Neural & Transformer modeling
    ├── 05_ensemble_ratios_nationwide.slurm    # Stage 05: Nationwide ensemble optimization
    ├── 05_ensemble_ratios_texas.slurm         # Stage 05: Texas ensemble optimization
    ├── 06_urban_rural_modeling.slurm          # Stage 06: Rural vs. Urban stratification
    ├── 07_condition_subgroups.slurm           # Stage 07: Comorbidity subgroup evaluation
    ├── 08_nationwide_models_benchmark.slurm   # Stage 08: Benchmark models comparison (NW)
    ├── 08_texas_models_benchmark.slurm        # Stage 08: Benchmark models comparison (TX)
    ├── 08_unified_parity_nationwide.slurm     # Stage 08: ACT-Parity optimization (NW)
    ├── 08_unified_parity_texas.slurm          # Stage 08: ACT-Parity optimization (TX)
    ├── 09_equity_experiments.slurm            # Stage 09: Comprehensive equity auditing
    ├── 10_reviewer_experiments.slurm          # Stage 10: Calibration, DCA & transport audits
    │
    ├── preprocessing_nb_condensed.py          # Nationwide data preprocessing pipeline
    ├── preprocessing_tx_condensed.py          # Texas statewide data preprocessing pipeline
    ├── preprocessing_orgOASIS.py              # OASIS raw feature transformations
    │
    ├── featureSelection/                      # Feature selection modules
    │   ├── feature_selection.py               # Multi-strategy feature selector (RF, MI, Lasso)
    │   ├── run_cohort_urban_rural_fs.py       # Subgroup feature selection runner
    │   └── utils.py                           # Shared FS utilities
    │
    ├── modeling/                              # Machine learning & transformer suite
    │   ├── models.py                          # GBDT, Tabular Transformer, HIR-M3 architectures
    │   ├── utils.py                           # Dataset loaders, cross-validation, batchers
    │   ├── metrics.py                         # Clinical discrimination & operating metrics
    │   ├── hier_icd_embedding.py              # Hierarchical ICD embedding layer (HICD-BERT)
    │   ├── optimize_ensemble.py               # Optimal ratio search
    │   ├── explore_ensemble_ratios.py         # Ratio grid-search driver
    │   ├── run_modeling.py                    # Baseline GBDT benchmark
    │   ├── run_neural_modeling.py             # Neural & Transformer benchmark
    │   ├── run_urban_rural_modeling.py        # Geographic stratification runner
    │   ├── run_condition_subgroup_modeling.py # 7 clinical comorbidity subgroup runner
    │   └── run_equity_experiments.py          # Baseline demographic equity audits
    │
    ├── parity/                                # Algorithmic fairness & ACT-Parity suite
    │   ├── models.py                          # ACTParityV2 neural architecture & heads
    │   ├── loss.py                            # Augmented Lagrangian Method (ALM) loss
    │   ├── metrics.py                         # Fairness evaluation routines
    │   ├── equity_metrics.py                  # Multi-attribute parity, FPSA, FNR-worst audits
    │   ├── compare_all_models_nationwide.py   # Full model parity evaluation (Nationwide)
    │   ├── compare_all_models_texas.py        # Full model parity evaluation (Texas)
    │   └── run_unified_parity_comparison.py   # Cross-model parity comparison driver
    │
    ├── scripts/                               # Statistical evaluation & figures
    │   ├── build_clean_datasets.py            # Cohort filtration & SDoH linkage
    │   ├── compute_table_cis.py               # 95% bootstrap confidence intervals
    │   ├── compute_hir_ablation_cis.py        # HIR-M3 ablation bootstrap CIs
    │   ├── eval_calibration.py                # Calibration deciles, ICI, ECE calculation
    │   ├── eval_dca.py                        # Decision Curve Analysis (Net Clinical Benefit)
    │   ├── eval_capacity.py                   # Clinical review capacity simulation
    │   ├── eval_transport.py                  # Cross-geographic transportability evaluation
    │   ├── run_reviewer_experiments.py        # Automated runner for reviewer analyses
    │   └── generate_publication_figures.py    # Vector graphic publication figure generator
    │
    ├── figures/                               # Attention tier figures & visual exploration
    └── results/                               # Benchmark metric outputs & calibration logs
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/AI2TXST/HIR-M3.git
cd HIR-M3

# Create and activate conda environment
conda create -n oasis_readmission python=3.10 -y
conda activate oasis_readmission

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
Review [`data/README.md`](data/README.md) for instructions on obtaining CMS OASIS and AHRQ SDoH datasets under a Data Use Agreement (DUA).

Place the prepared datasets in the `data/` directory:
- `data/processed_final_mergedDF_condensed.csv` (Nationwide cohort)
- `data/processed_final_mergedDF_condensed_TX.csv` (Texas cohort)

### 3. Pipeline Execution (SLURM / HPC)

Navigate to the `src/` directory to run the pipeline scripts:
```bash
cd src

# Master execution of all 10 stages sequentially
bash run_all_slurm_pipeline.sh
```

Or submit individual stages from the repository root:
```bash
# Stage 01: Feature selection
sbatch src/02_feature_selection.slurm

# Stage 03: Baseline modeling
sbatch src/03_primary_modeling.slurm

# Stage 08: ACT-Parity optimization
sbatch src/08_unified_parity_nationwide.slurm
sbatch src/08_unified_parity_texas.slurm

# Stage 10: Reviewer analyses (Calibration, DCA, Transportability)
sbatch src/10_reviewer_experiments.slurm
```

---

## 📊 Key Methodological Contributions

1. **Social-Ecological Hierarchical Modeling (HIR-M3)**: Organizes predictors into Micro (clinical/functional), Meso (tract SDoH/ADI), and Macro (facility case-mix) tiers with gated attention and ICD-10 hierarchy embeddings.
2. **Clinical Asymmetry in Algorithmic Fairness**: Formulates parity constraints around False Negative Rate disparities ($\text{FNR}_{\text{worst}}$) to prevent under-detection of high-risk patients.
3. **Augmented Lagrangian Constrained Optimization (ACT-Parity)**: Enforces fairness bounds dynamically via dual multipliers and quadratic penalties, avoiding gradient collapse and arbitrary trade-offs.
4. **Continuous-Threshold Equity Auditing (FPSA)**: Evaluates fairness-performance stability across 41 decision thresholds ($0.10 \le \tau \le 0.50$) rather than static cutoffs.
5. **Cross-Geographic Transport Auditing**: Measures empirical calibration drift and equity transport gaps ($\Delta$) when transferring national models to regional state deployments.

---

## 📜 Compliance and Checklist

- **TRIPOD-AI Checklist**: Complete item-by-item compliance is documented in [`docs/TRIPOD_AI_CHECKLIST_AND_S1_MATRIX.md`](docs/TRIPOD_AI_CHECKLIST_AND_S1_MATRIX.md).
- **Supplementary Materials**: Extended tables, derivations, and full hyperparameter search grids are available in [`docs/supplementary_materials.tex`](docs/supplementary_materials.tex).

---

## 📄 Citation

```bibtex
@article{oasis_HIR_M3_2026,
  title   = {{HIR-M3}: Hierarchy-Aware Tabular Modeling and {ACT-Parity} Evaluation for 30-Day Acute-Care Utilization After Home Health Care},
  author  = {Elizondo, Mirna and Te{\v{s}}i{\'c}, Jelena},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review}
}
```

---

## ⚖️ License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
