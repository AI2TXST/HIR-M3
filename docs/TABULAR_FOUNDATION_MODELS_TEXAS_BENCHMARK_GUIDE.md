# Tabular Foundation Models (TabPFN, TabICL) vs. Domain & Baseline Architectures: Texas Cohort Benchmark Guide

**Target Script**: [`parity/compare_all_models_texas.py`](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/parity/compare_all_models_texas.py)  
**Cluster SLURM Job**: [`08_texas_models_benchmark.slurm`](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/08_texas_models_benchmark.slurm)  
**Target Dataset**: 100% Texas OASIS Cohort ($N = 62,449$, Features $D = 756$)  
**Evaluation Framework**: Health Equity and Algorithmic Learning (HEAL) & Clinical Readmission Risk Auditing

---

## 1. Overview & Purpose

This pipeline provides an empirical benchmark comparing emerging **Tabular Foundation Models (TFMs)** (specifically **TabPFN** and **TabICL**) against the full continuum of predictive models developed for 30-day home healthcare readmission risk:

1. **Classical Baselines**: Logistic Regression, Random Forest
2. **Gradient Boosted Decision Trees (GBDTs)**: LightGBM, XGBoost, CatBoost
3. **Deep Tabular Architectures**: Multi-Layer Perceptron (MLP)
4. **Domain-Hierarchical Networks**: HIR-M3 Transformer & 70:30 XGBoost-HIR-M3 Ensemble
5. **Fairness-Constrained Architectures**: ACT-Parity v2 & HIR-M3 + ACT-Parity Hybrid
6. **Tabular Foundation Models (TFMs)**: TabPFN (Zero-Shot Prior-Fitted Network), TabICL (In-Context Meta-Learner), and TabFM (Feature-Tokenized Transformer Backbone)

---

## 2. Architectural Comparison Across Model Tiers

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    MODEL SPECTRUM & PARADIGMS                                    │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
   [Traditional & Tree]             [Domain-Hierarchical]             [In-Context Foundational]
  • Logistic Regression            • HIR-M3 Transformer             • TabPFN (Zero-Shot)
  • Random Forest                  • ACT-Parity v2 (D-GAP)           • TabICL (In-Context)
  • XGBoost / CatBoost / LGBM      • Multi-Tier Token Gating         • Prior-Fitted Transformers
```

| Dimension | Classical & GBDTs | HIR-M3 & ACT-Parity v2 | TabPFN / TabICL (TFMs) |
| :--- | :--- | :--- | :--- |
| **Model Weights** | Trained via ERM on Texas data | Trained with cross-attention + D-GAP loss | **Pre-trained offline** on synthetic priors |
| **Training Paradigm** | Gradient descent / tree splitting | End-to-end backpropagation | **In-context inference** (No weight updates) |
| **Feature Dimension ($D$)** | Full feature set ($D = 756$) | Micro, Meso, Macro tokens ($D = 756$) | **Budget-limited** ($D \le 100$) |
| **Context Sample Size ($N$)**| Full train split ($N \approx 45{,}000$) | Full train split ($N \approx 45{,}000$) | **Subsampled context** ($N = 5{,}000$) |
| **ICD Hierarchy Handling** | Flat binary flags | Hierarchical ICD-10 embeddings | Unstructured flat columns |
| **Fairness Optimization** | None (Post-hoc thresholding) | **Differentiable D-GAP Lagrangian** | None (Zero-shot posterior audit) |

---

## 3. What Had to Be Changed / Altered for Tabular Foundation Models

Tabular Foundation Models were designed for general small-to-medium benchmark datasets (e.g., OpenML). Ingesting large-scale, high-dimensional healthcare data required several technical modifications:

### A. Context Length Subsampling ($N = 62,449 \to 5,000$)
* **The Problem**: TabPFN and TabICL are in-context transformers that ingest training samples as prompt tokens. An attention matrix over $45,000$ training rows scales quadratically ($O(N^2)$), causing GPU CUDA Out-Of-Memory (OOM) crashes.
* **The Fix**: The `TabPFNWrapper` uses `StratifiedShuffleSplit` to extract a balanced, highly representative support set of $N = 5{,}000$ samples from the training split.

### B. Feature Dimensionality Budgeting ($D = 756 \to 100$)
* **The Problem**: TabPFN hard-caps feature dimensionality at $D \le 100$ (v1) or $D \le 500$ (v2). Passing all 756 raw OASIS columns results in dimension errors.
* **The Fix**: The wrapper dynamically selects the top 100 highest-variance clinical and structural features for the TFM context window, while GBDTs and HIR-M3 continue to receive the complete 756-feature space.

### C. Chunked Test-Time Inference Batching ($N_{\text{test}} \approx 9,367$)
* **The Problem**: Querying $9,367$ test patients simultaneously against a $5,000$-sample context set overwhelms GPU VRAM during the cross-attention forward pass.
* **The Fix**: `predict_proba` chunks the test queries into $2,000$-sample batches, computes posteriors sequentially on the GPU, and stacks them into a unified probability vector.

### D. Data Type Sanitization & Missing Value Imputation
* **The Problem**: TFMs cannot process string categoricals or `NaN`/`Inf` floating-point anomalies.
* **The Fix**: Coerces all features with `pd.to_numeric(errors='coerce').fillna(0.0)` followed by `StandardScaler` normalization to `float32`.

### E. Handling Imbalanced Readmission Distributions
* **The Problem**: The OASIS readmission positive rate is $\sim 15\text{--}20\%$. TabPFN's synthetic prior assumes relatively balanced class distributions.
* **The Fix**: `RandomOverSampler` is applied to balance the training pool prior to context subsampling, ensuring the in-context prompt presents balanced class evidence to the model.

---

## 4. Evaluation Protocol & Audit Metrics

All models are evaluated on the exact same held-out test split ($15\%$ test set) using optimal decision thresholds determined by maximizing the F1-Score on the Precision-Recall curve:

### 1. Discriminative Performance
* **ROC-AUC** & **PR-AUC**: Overall ranking capability and precision-recall trade-off.
* **F1-Score, Precision, Sensitivity (TPR), Accuracy**: Thresholded classification utility.
* **AUC Retention Ratio**: Fraction of baseline discrimination retained relative to XGBoost ($AUC / AUC_{\text{XGB}}$).

### 2. Clinical Calibration
* **Brier Score**: Mean squared error between predicted probabilities and binary outcomes.
* **Platt Calibration Slope**: Slope of the logistic calibration curve (ideal slope $\in [0.9, 1.1]$).

### 3. Algorithmic Equity & Subgroup Parity
* **Worst-Group FNR ($\text{FNR}_{\text{worst}}$)**: Highest False Negative Rate across racial/ethnic groups (Black, Hispanic, Asian, AIAN, NHPI, White).
* **FNR Gap ($\Delta \text{FNR}$)**: $\text{FNR}_{\text{worst}} - \text{FNR}_{\text{best}}$.
* **Equalized Odds Difference**: Maximum disparity across True Positive and False Positive rates.
* **Harm-Weighted Excess FNR Index ($\text{EFNHI}^*$)**: Penalizes disproportionate missed readmissions in vulnerable populations.

---

## 5. Execution Instructions

### A. Run via SLURM on HPC Cluster
```bash
sbatch 08_texas_models_benchmark.slurm
```

### B. Run Locally or Interactively
```bash
cd parity
python compare_all_models_texas.py
```

### C. Output Deliverables
* **CSV Data Table**: `parity/results/texas_all_models_comparative_benchmark.csv`
* **Markdown Summary Report**: `docs/TEXAS_ALL_MODELS_COMPARATIVE_REPORT.md`
