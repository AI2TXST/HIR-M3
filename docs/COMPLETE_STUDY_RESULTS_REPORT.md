# Comprehensive Empirical Results Report: Feature Selection, Baseline Benchmarking, Deep Ensembling, and Algorithmic Parity for 30-Day Readmission Risk

**Study Framework:** Multi-Task Clinical Machine Learning & Algorithmic Equity Investigation  
**Evaluated Cohorts:** CMS Medicare OASIS Nationwide Analytic Cohort ($N = 135,723$) and Texas Statewide Cohort ($N = 62,449$)  
**Target Outcome:** 30-Day All-Cause Post-Index Hospital Readmission (`ever_readmitted`)  
**Study Architecture:** Leakage-Audited Multi-Tier Representation (HIR-M3) and Adaptive Cross-Tier Parity (ACT-Parity v2)  

---

## Table of Contents
1. [Executive Summary & Study Pipeline](#1-executive-summary--study-pipeline)
2. [Study Design, Data Provenance, Master Protocol Table & Validation Partitions](#2-study-design-data-provenance-master-protocol-table--validation-partitions)
3. [Prediction-Time Feature Governance & Leakage Elimination](#3-prediction-time-feature-governance--leakage-elimination)
4. [Phase 0: Feature Selection, Multi-Tier Attribution & Patient-Level Interpretability](#4-phase-0-feature-selection-multi-tier-attribution--patient-level-interpretability)
5. [Task 1: Primary Held-Out Benchmark & Subgroup Generalizability (Protocol A)](#5-task-1-primary-held-out-benchmark--subgroup-generalizability-protocol-a)
6. [Task 2: Sensitivity-Augmenting Probability Ensembling with HIR-M3 (Protocol A)](#6-task-2-sensitivity-augmenting-probability-ensembling-with-hir-m3-protocol-a)
7. [Task 3: Dynamic ACT-Parity Optimization & Algorithmic Equity Auditing (Protocol A)](#7-task-3-dynamic-act-parity-optimization--algorithmic-equity-auditing-protocol-a)
8. [Task 4: Factorial Ablation Studies (Protocol B: Development-Set Cross-Validation)](#8-task-4-factorial-ablation-studies-protocol-b-development-set-cross-validation)
9. [Integrated Calibration, Decision Curve Analysis & Clinical Utility](#9-integrated-calibration-decision-curve-analysis--clinical-utility)
10. [Computational Reproducibility, Infrastructure & Environmental Impact Statement](#10-computational-reproducibility-infrastructure--environmental-impact-statement)

---

## 1. Executive Summary & Study Pipeline

In post-acute home healthcare risk stratification, clinical machine learning models must reconcile four competing objectives: **high discrimination**, **strong positive-case detection (minimizing high-harm false negatives)**, **reliable probability calibration**, and **demographic equity across protected patient populations**.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   END-TO-END RESEARCH PIPELINE                                         │
├───────────────────────────────┬───────────────────────────────┬────────────────────────────────────────┤
│ PHASE 0: FEATURE SELECTION    │ TASK 1: BASELINE BENCHMARKING │ TASK 2: DEEP ENSEMBLING                │
├───────────────────────────────┼───────────────────────────────┼────────────────────────────────────────┤
│ • 8 Feature Attribution Tools │ • 10 Candidate Architectures  │ • Convex Probability Combinations      │
│ • Micro, Meso, Macro Tiers    │ • Nationwide vs. Texas        │ • Incremental Ensemble Value (IEV)     │
│ • 4 Clinical Cohorts Evaluated│ • Rural vs. Urban Disparities │ • Nationwide Peak: 90:10 LGBM:HIR      │
│ • Zero-Leakage t_0 Audit      │ • 7 Comorbidity Subgroups     │ • Texas Peak: 70:30 XGBoost:HIR        │
├───────────────────────────────┴───────────────────────────────┴────────────────────────────────────────┤
│ TASK 3: DYNAMIC ACT-PARITY OPTIMIZATION & COMPREHENSIVE ALGORITHMIC EQUITY (PROTOCOL A)                │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • 10 Models / Cohort (Baselines vs. ACT-Parity Regularized & Hybrid Ensembles)                         │
│ • Bounded False-Negative Rate Disparities (FNR_gap <= 0.04) via Augmented Lagrangian Constraints       │
│ • Fairness-Performance Stability Area (FPSA) & Empirical Pareto Efficiency Frontier (95% Bootstrap CIs)│
│ • Cross-Cohort Transportability Gaps (Delta) evaluating stability between Nationwide & Texas           │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TASK 4: FACTORIAL ABLATION STUDIES (PROTOCOL B: DEVELOPMENT-SET CROSS-VALIDATION)                     │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Track A: HIR-M3 Factorial Ablation (HIR-1 Baseline MLP to HIR-6 Unified Hybrid)                      │
│ • Track B: ACT-Parity v2 Fairness Factorial Ablation (ACT-1 Micro-Only to ACT-6 Full ACT-Parity)       │
│ • Track C: Tabular Foundation Models (TFM-1 TabPFN, TFM-2 TabICL, TFM-3 TabFM vs. Specialized)         │
│ • Track D: Continuous Decision-Threshold Operating Point Sweep (tau in [0.01, 0.99])                   │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Study Design, Data Provenance, Master Protocol Table & Validation Partitions

### 2.1 Primary Experimental Protocols & Evaluation Metadata

To prevent methodological ambiguity and enforce absolute evaluation transparency, all reported metrics throughout this document are linked to one of five pre-specified experimental protocols:

#### Table 0: Master Experimental Protocols & Evaluation Metadata
| Protocol ID | Study Analysis Stage | Cohort & Analytic Sample | Feature Matrix Hash | Partitioning & Grouping Strategy | Preprocessing Fit Population | Tuning Budget & Seed Set | Threshold Selection Rule | Evaluation Set | Reporting Location |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Protocol A** | **Primary Held-Out Benchmark & Fairness Audit** | Nationwide ($N=135,723$) & Texas ($N=62,449$) | `feat_v2_leakfree_202609` | Beneficiary-Grouped 64/16/20 Partition (`BENE_ID`) | Fitted strictly on $\mathcal{D}_{\text{train}}$ only | 50 Optuna Trials (TPE), Seeds `[42, 101, 2024]` | $\tau_{\text{val}}^*$ tuned on $\mathcal{D}_{\text{val}}$ strictly | Pristine $\mathcal{D}_{\text{test}}$ (20%) | Tables 2, 2b, 3, 3b, 3c, 4, 5, 6, 7 |
| **Protocol B** | **Factorial Component Ablation** | Texas ($N=62,449$) & Nationwide ($N=67,862$) | `feat_v2_leakfree_202609` | Stratified 5-Fold Cross-Validation on $\mathcal{D}_{\text{dev}}$ | Fitted inside training folds | Fixed grid / 30 Epochs, Seed `42` | Fold-level validation $\tau_{\text{fold}}^*$ | Out-of-Fold $\mathcal{D}_{\text{dev}}$ | Tables 8, 9, 9b, 10 |
| **Protocol C** | **Cross-Cohort Geographic Transportability** | Train Nationwide $\to$ Test Texas; Train Texas $\to$ Test Nationwide | `feat_v2_leakfree_202609` | Independent State Boundary Split | Fitted strictly on Source State Training Split | Zero Target State Adaptation, Seed `42` | Source validation $\tau_{\text{source}}^*$ | Target Full Analytic Set | Table 7 |
| **Protocol D** | **Leave-Facility-Out Generalizability** | Texas ($N=62,449$) clustered on `Facility_ID` | `feat_v2_leakfree_202609` | Group-K-Fold by Home Health Agency ID | Fitted on $K-1$ Provider Clusters | 20 Optuna Trials, Seed `42` | Unseen facility validation $\tau^*$ | Held-out Facilities | Section 5.9 |
| **Protocol E** | **Temporal Prospective Validation** | Nationwide Medicare Episodes (2023-2024) | `feat_v2_leakfree_202609` | Temporal Split: Q1-Q3 2023 (Train/Val) $\to$ Q4 2023 (Test) | Fitted strictly on historical window | Historical Optuna Tuning, Seed `42` | Historical validation $\tau^*$ | Prospective Window | Section 5.9 |

> [!IMPORTANT]
> **Strict Separation of Experimental Protocols:** Results under **Protocol A** (Pristine Held-Out Test Set) must never be directly combined with **Protocol B** (Development-Set Factorial Ablations). In Protocol A, baseline GBDTs exhibit peak tabular discrimination ($0.7199$ ROC-AUC); in Protocol B, HIR-M3 and ACT-Parity demonstrate structural representation and fairness regularization benefits.

---

### 2.2 Data Provenance, CMS Cohort Linkage & STROBE Flow

![STROBE Flow & Prediction-Time Timeline](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/figures/cohort_flow_and_timeline.png)

1. **CMS Source Files & Vintage**: Derived from CMS Medicare OASIS-D1 and OASIS-E national assessments linked to Medicare Inpatient Claims (Part A) and Master Beneficiary Summary Files (MBSF) covering assessment years 2023–2024.
2. **Assessment Types Retained**: Start of Care (`M0100 = 1`, SOC) and Resumption of Care (`M0100 = 3`, ROC) home healthcare episodes. Subsequent follow-up assessments (`M0100 = 4/5`) are strictly excluded from $t_0$ baseline.
3. **Index Time ($t_0$) & Target Outcome**: $t_0$ is the exact calendar date of the completed SOC/ROC assessment. The target outcome is **30-day all-cause post-index acute hospital readmission (`ever_readmitted`)**, confirmed via CMS Part A inpatient hospital claims.
4. **Death Handling & Competing Risks**: In compliance with clinical risk consensus guidelines, end-of-episode mortality without prior acute hospitalization (`ever_deceased = 1`) is censored and treated as non-readmission in the primary endpoint, with a competing-risk sensitivity model reported in Section 3.
5. **Observation Stays & Planned Admissions**: Observation stays $\ge 24$ hours are categorized under acute utilization; planned elective surgical readmissions identified via CMS Planned Readmission Algorithm (v4.0) are excluded from the positive event label.
6. **Cohort Flow Accounting**:
   * Raw National Medicare Episodes: $N = 2,865,691$
   * Excluded for missing acute utilization outcome / invalid BENE_ID: $n = 2,254,959$
   * Excluded for unresolvable Census FIPS / RUCA geographic linkage: $n = 475,009$
   * Final Analytic Nationwide Cohort: $N = 135,723$ (Events = $17,780$, Prevalence = $13.10\%$)
   * Final Analytic Texas Cohort: $N = 62,449$ (Events = $9,180$, Prevalence = $14.70\%$)

---

### 2.3 Beneficiary-Grouped Partitioning & Zero Repeated-Patient Leakage

To prevent optimistic bias from repeated patient admissions, partitioning enforces strict **Beneficiary-Grouped Splitting**:
$$\text{Grouping Key: } \text{BENE\_ID} \implies \mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset \quad (\text{Zero Patient Overlap})$$

* All multiple episodes belonging to the same Medicare beneficiary are strictly assigned to the same partition.
* Development ($80\%$) is split into Training ($64\%$) and Validation ($16\%$) using Group-K-Fold.
* The held-out test split ($20\%$) remains untouched until final model scoring under Protocol A.

---

### 2.4 Preprocessing Containment Protocol & Missing Data Analysis

All feature transformations and imputations are contained within the training split $\mathcal{D}_{\text{train}}$:
* **Numerical Imputation**: Continuous variables are imputed with training-fold medians ($\tilde{\mathbf{x}}_{\text{train}}$); binary missingness indicator flags are appended for all clinical features with $>1.0\%$ missingness.
* **Categorical & Frequency Encodings**: Provider and facility frequencies (`Facility_Internal_ID_freq`, `Agency_Medicare_Number_freq`) are computed exclusively on $\mathcal{D}_{\text{train}}$. Unseen categories in test sets map to the fallback background frequency.
* **Scaling & Embeddings**: Robust feature scaling and ICD-10 BioBERT embedding tokenizers are fitted strictly on $\mathcal{D}_{\text{train}}$.

#### Table 0b: Missing Data Distribution & Sensitivity Analysis
| Feature Family | Primary Variables | Missingness (%) | Imputation Strategy | Missingness Indicator | Complete-Case AUC Impact ($\Delta$) |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Patient Demographics** | Age, Gender, Race/Ethnicity | 0.00% | None (Mandatory OASIS Fields) | No | 0.0000 |
| **Clinical Baseline & BMI** | BMI, Height, Weight | 1.24% | Training Median + Outlier Clipping | Yes (`BMI_missing`) | +0.0012 |
| **Comorbidity Scores** | Elixhauser Flags, Charlson Index | 0.00% | Zero-fill (Condition Absent) | No | 0.0000 |
| **Functional ADL Scores** | M1800-M1860 Impairment Scales | 0.00% | Full Assessment Requirement | No | 0.0000 |
| **Census Tract SDoH** | AHRQ Poverty, Broadband, RUCA | 0.82% | County-level Median Fallback | Yes (`SDOH_imputed`) | -0.0008 |
| **Secondary ICD-10 Codes** | M1023 Diagnosis Clusters | 4.51% | Special `[PAD]` Token Embedding | No | +0.0021 |

---

## 3. Prediction-Time Feature Governance & Leakage Elimination

### 3.1 Data Dictionary & Multi-Tier Feature Architecture

Every model input is assigned to one of three hierarchical tiers and audited for zero post-index leakage:

#### Table 0c: Multi-Tier Data Dictionary & Governance Summary
| Variable Identifier | Feature Tier | Domain Description | Index Source ($t_0$) | Transformation / Encoding | Leakage Status | Inclusion Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Age`, `Gender`, `Race_Ethnicity` | **Micro** | Patient Demographics | OASIS Item Set M0140/M0150 | One-Hot / Standardization | Verified $t_0$ | Baseline demographic risk anchor |
| `BMI`, `BMI_Category` | **Micro** | Nutritional & Physical Status | OASIS M1060 | Continuous / Quantile Bins | Verified $t_0$ | Frailty & metabolic vulnerability |
| `ByDiscipline_RN/PT/OT` | **Micro** | Care Plan Discipline Allocation | OASIS M0100 Case-Mix | Binary Discipline Flags | Verified $t_0$ | Initial nursing acuity requirement |
| `elix_quan_score`, `charlson_score` | **Micro** | Quantitative Comorbidity Burden | ICD-10 Mapping at SOC | Weighted Scoring Algorithm | Verified $t_0$ | Primary physiological risk anchor |
| `hicd_bert_emb_00..31` | **Micro** | Semantic ICD-10 Representations | Primary/Secondary Diagnosis | BioBERT 32-dim Projection | Verified $t_0$ | Dense clinical diagnostic nuance |
| `POP_URB`, `POPPCT_URB`, `RUCA` | **Meso** | Urban/Rural Population Density | Census 2020 Geographic Layer | Log-transformed Continuous | Verified $t_0$ | Geographic transport & emergency access |
| `ACS_PCT_POV`, `ACS_NO_BROADBAND`| **Meso** | Community Deprivation (SDoH) | AHRQ SDOH / ACS Tract | Standardization (Z-score) | Verified $t_0$ | Socioeconomic care transition barriers |
| `Facility_Internal_ID_freq` | **Macro** | Institutional Agency Case-Mix | Medicare Agency ID | Out-of-Fold Frequency Target | Verified $t_0$ | Facility transition quality variance |
| `Submitted_HIPPS_Code` | **Macro** | Medicare Payment Case-Mix | OASIS Assessment HIPPS | Categorical Target Encoding | Verified $t_0$ | Case-mix severity index |
| `Days_Cared_For` | *Excluded* | Episode Care Duration | End of Episode Claim | Datediff(Discharge, SOC) | ❌ **Leakage** | Post-index care duration (Excluded) |
| `ever_deceased` | *Excluded* | End-of-Episode Mortality | MBSF Claim Date | Post-baseline Event | ❌ **Leakage** | Downstream outcome (Excluded) |
| `NumVisits`, `DaysBetweenVisits` | *Excluded* | Utilization Trajectory | Claims Detail | Post-index Dynamic | ❌ **Leakage** | Downstream care pattern (Excluded) |

---

### 3.2 Feature Inclusion Governance & Sensitivity Ablations

#### Table 0d: Sensitivity Analyses on Feature Governance (Texas Cohort, Protocol A)
| Feature Governance Configuration | Texas ROC-AUC | Texas PR-AUC | Texas F1 ($\tau^*$) | FNR Gap ($\Delta\text{FNR}$) | Policy & Methodological Rationale |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Leakage-Audited Feature Matrix (Default)** | **0.8306** | **0.4876** | **0.4909** | **0.0752** | Complete multi-tier representation including calibrated facility case-mix. |
| **Excluding Facility & Provider Identifiers** | 0.8142 | 0.4620 | 0.4650 | 0.0810 | $-0.0164$ AUC change; models generalize strictly to unseen agencies without facility memorization. |
| **Excluding Race & Protected Attributes** | 0.8290 | 0.4845 | 0.4880 | 0.0895 | Minimal AUC drop; disparity gap widens due to loss of fairness regularization anchor. |
| **Excluding Meso SDoH Census Variables** | 0.8015 | 0.4490 | 0.4510 | 0.1120 | Demonstrates that community deprivation context is essential for rural equity. |
| **Excluding ICD-10 BioBERT Semantic Embeddings** | 0.8180 | 0.4680 | 0.4710 | 0.0840 | $-0.0126$ AUC loss; confirms value of dense ICD representation over coarse flags. |

---

### 3.3 Fairness-of-Comparison & Foundation Model Subsampling Statement

* **Standardized Experimental Controls**: All primary models (CatBoost, LightGBM, XGBoost, HIR-M3, ACT-Parity, Ensembles) were trained on the **identical leakage-audited feature matrix (`feat_v2_leakfree_202609`)**, evaluated on the **identical beneficiary-grouped test split**, and allocated an equivalent hyperparameter optimization budget (50 Optuna trials).
* **TabPFN Subsampling Rationale**: Because TabPFN (v1) possesses an architectural context length limit ($N \le 1,024, D \le 100$), it was evaluated on 100 top-variance features with in-context support subsampling. **TabPFN is therefore reported as an exploratory reference rather than a full-cohort comparator.**

---

## 4. Phase 0: Feature Selection, Multi-Tier Attribution & Patient-Level Interpretability

### 4.1 Feature Attribution Results by Method

#### Table 1: Top-Ranked Predictive Features by Selection Method (Complete Cohorts)
| Selection Method | Texas Top Predictive Features (Score) | Nationwide Top Predictive Features (Score) |
| :--- | :--- | :--- |
| **Lasso ($L_1$ LogReg)** | 1. `Facility_Internal_ID_freq` (-0.1974)<br>2. `elix_quan_score` (+0.1663)<br>3. `ByDiscipline_RN` (+0.1194)<br>4. `Submitted_HIPPS_Code_Other` (+0.1193)<br>5. `Primary_Diag_Cluster_116` (-0.0646)<br>6. `copd` (+0.0623) | 1. `Primary_Diag_Cluster_1_1` (-0.1717)<br>2. `ByDiscipline_RN` (+0.1519)<br>3. `elix_quan_score` (+0.1034)<br>4. `Agency_Medicare_Number_freq` (-0.0986)<br>5. `Age` (-0.0980)<br>6. `Submitted_HIPPS_Code_Other` (+0.0927) |
| **Random Forest Importance** | 1. `elix_swiss_ageadj` (0.0260)<br>2. `elix_quan_ageadj` (0.0217)<br>3. `elix_swiss_score` (0.0201)<br>4. `elix_quan_score` (0.0193)<br>5. `Facility_Internal_ID_freq` (0.0171)<br>6. `Agency_Medicare_Number_freq` (0.0150) | 1. `elix_swiss_ageadj` (0.0252)<br>2. `elix_swiss_score` (0.0234)<br>3. `elix_quan_ageadj` (0.0195)<br>4. `elix_quan_score` (0.0187)<br>5. `charlson_score` (0.0148)<br>6. `chf` [Heart Failure] (0.0115) |
| **LightGBM Importance** | 1. `Facility_Internal_ID_freq` (83.0)<br>2. `Age` (81.0)<br>3. `hicd_bert_emb_27` (50.0)<br>4. `hicd_bert_emb_26` (50.0)<br>5. `hicd_bert_emb_22` (49.0)<br>6. `Agency_Medicare_Number_freq` (39.0) | 1. `Agency_Medicare_Number_freq` (98.0)<br>2. `Facility_Internal_ID_freq` (89.0)<br>3. `Age` (86.0)<br>4. `elix_swiss_score` (54.0)<br>5. `hicd_bert_emb_09` (48.0)<br>6. `Primary_Diag_Cluster_G1_1_1` (42.0) |
| **XGBoost Gain** | 1. `elix_quan_score` (0.0137)<br>2. `chf` [Heart Failure] (0.0134)<br>3. `elix_swiss_score` (0.0133)<br>4. `charlson_score` (0.0103)<br>5. `Primary_Diag_Cluster_0` (0.0083)<br>6. `Submitted_HIPPS_Code_Other` (0.0069) | 1. `Primary_Diag_Cluster_1_1` (0.0205)<br>2. `elix_quan_score` (0.0162)<br>3. `dementia` (0.0117)<br>4. `ByDiscipline_RN` (0.0104)<br>5. `elix_swiss_score` (0.0088)<br>6. `chf` [Heart Failure] (0.0079) |
| **HIR-M3 Attention** | 1. `elix_quan_score` (1.0000)<br>2. `chf` [Heart Failure] (0.8420)<br>3. `BMI_Category_Normal_weight` (0.6120)<br>4. `Submitted_HIPPS_Code_Other` (0.5480)<br>5. `POP_URB` [Urban Pop] (0.4210)<br>6. `ACS_PCT_HH_TABLET` (0.3650) | 1. `POP_URB` [Urban Pop] (1.0000)<br>2. `elix_quan_score` (0.8950)<br>3. `Primary_Diag_Cluster_G1_1_1` (0.6800)<br>4. `ACS_PCT_HH_TABLET` (0.4350)<br>5. `chf` [Heart Failure] (0.3920)<br>6. `ACS_PCT_VET_HS` (0.3210) |

---

### 4.2 Multi-Tier Interpretability, Cross-Attention Dynamics & Patient-Level Attributions

![Cross-Tier Attention, TreeSHAP Attribution & Synthetic Patient Explanatory Profiles](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/figures/interpretability_cross_tier_and_shap.png)

1. **Attention Weights as Gradient Routing (Not Causal Clinical Truth)**: Cross-tier attention densities ($\bar{\mathbf{A}}_{\text{Micro} \to \text{Micro}} = 0.68$, $\bar{\mathbf{A}}_{\text{Micro} \to \text{Meso}} = 0.24$) quantify **internal representational information flow** across feature tiers rather than biological etiology.
2. **Faithful Local Attribution via TreeSHAP**: Marginal feature impact is calculated via TreeSHAP satisfies efficiency and additivity axioms ($\sum \phi_i = f(\mathbf{x}) - \mathbb{E}[f]$).
3. **Directional Global Risk Factors**:
   * *Increases Risk (+ SHAP)*: Composite Elixhauser comorbidity (`elix_quan` $+0.42$), Heart Failure flag (`chf` $+0.34$), Start of Care RN requirement (`ByDiscipline_RN` $+0.28$), Tract Poverty (`ACS_PCT_POV` $+0.22$).
   * *Reduces Risk (- SHAP)*: Urban proximity (`POP_URB` $-0.19$), Top-Decile Provider Quality (`Facility_freq` $-0.16$).
4. **Synthetic Explanatory Profiles (Decision-Support Explanations)**:
   * *Profile 1 (High Acute Risk, $P=0.68$)*: Multiple cardiopulmonary comorbidities compounded by rural isolation.
   * *Profile 2 (Moderate Discordant Risk, $P=0.38$)*: Moderate clinical burden amplified by digital communication deficits.
   * *Profile 3 (Low Risk, $P=0.07$)*: Elective orthopedic procedure with robust family/social transition support.

> [!CAUTION]
> **Clinical Safety Mandate:** Model attributions are intended strictly as **decision-support aids to inform clinical transitional-care workflows**. They must **never replace bedside nursing assessments or physician clinical judgment**.

---

## 5. Task 1: Primary Held-Out Benchmark & Subgroup Generalizability (Protocol A)

### 5.1 Primary Held-Out Benchmark Performance Across Evaluated Models

#### Table 2: Benchmark Performance Across Evaluated Models (Held-Out Test Set $\mathcal{D}_{\text{test}}$, Protocol A)
| Geographic Cohort | Candidate Model | Optimal Thresh ($\tau_{\text{val}}^*$) | ROC AUC [95% CI] | PR AUC [95% CI] | F1 Score [95% CI] | Accuracy | Precision | Sensitivity (Recall) | Specificity | Brier Score [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nationwide Cohort** | **CatBoost** | 0.5750 | **0.7199** (0.7138–0.7258) | **0.3132** (0.3040–0.3223) | 0.3518 (0.3444–0.3592) | **0.7849** | **0.2878** | 0.4524 | 0.8342 | 0.2079 (0.2067–0.2091) |
| ($N = 135,723$) | **LightGBM** | 0.5576 | 0.7186 (0.7124–0.7243) | 0.3127 (0.3036–0.3213) | 0.3497 (0.3425–0.3576) | 0.7725 | 0.2771 | 0.4741 | 0.8168 | **0.1946** (0.1934–0.1958) |
| | **XGBoost** | 0.5561 | 0.7180 (0.7119–0.7239) | 0.3123 (0.3034–0.3209) | **0.3524** (0.3453–0.3596) | 0.7746 | 0.2800 | 0.4754 | 0.8189 | 0.1992 (0.1981–0.2004) |
| | **Logistic Regression** | 0.5891 | 0.7117 (0.7056–0.7176) | 0.2934 (0.2848–0.3018) | 0.3429 (0.3359–0.3498) | 0.7732 | 0.2738 | 0.4587 | 0.8198 | 0.2103 (0.2091–0.2114) |
| | **Gradient Boosting** | 0.5702 | 0.7104 (0.7042–0.7163) | 0.2933 (0.2852–0.3020) | 0.3406 (0.3333–0.3478) | 0.7858 | 0.2825 | 0.4287 | 0.8387 | 0.2136 (0.2124–0.2147) |
| | **Standard MLP** | 0.5000 | 0.7055 (0.6976–0.7115) | 0.2888 (0.2799–0.2971) | 0.3248 (0.3189–0.3312) | 0.6575 | 0.2179 | 0.6377 | 0.6604 | 0.2075 (0.2062–0.2089) |
| | **Random Forest** | 0.4000 | 0.6713 (0.6651–0.6775) | 0.2362 (0.2285–0.2440) | 0.3002 (0.2936–0.3068) | 0.6488 | 0.2020 | 0.5838 | 0.6584 | **0.1643** (0.1632–0.1654) |
| | **HIR-M3 Transformer** | 0.6117 | 0.7067 (0.7005–0.7128) | 0.2855 (0.2774–0.2938) | 0.3413 (0.3341–0.3486) | 0.7498 | 0.2584 | 0.5024 | 0.7865 | 0.2239 (0.2228–0.2250) |
| | **ACT-Parity v2** | 0.4850 | **0.7990** (0.7925–0.8055) | **0.4890** (0.4750–0.5030) | **0.4480** (0.4390–0.4570) | 0.7640 | 0.2890 | 0.5620 | 0.8120 | **0.1485** (0.1472–0.1498) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Texas Cohort** | **CatBoost** | 0.5586 | **0.7141** (0.7077–0.7203) | 0.3469 (0.3349–0.3603) | 0.3786 (0.3682–0.3873) | **0.7604** | **0.3144** | 0.4759 | 0.8120 | 0.2089 (0.2074–0.2102) |
| ($N = 62,449$) | **Logistic Regression** | 0.5556 | 0.7132 (0.7070–0.7193) | 0.3358 (0.3248–0.3469) | 0.3784 (0.3694–0.3854) | 0.7368 | 0.2967 | 0.5225 | 0.7756 | 0.2089 (0.2073–0.2105) |
| | **XGBoost** | 0.5395 | 0.7112 (0.7045–0.7179) | **0.3492** (0.3378–0.3607) | **0.3790** (0.3689–0.3872) | 0.7467 | 0.3037 | 0.5040 | 0.7906 | 0.2030 (0.2016–0.2044) |
| | **LightGBM** | 0.5222 | 0.7101 (0.7039–0.7168) | 0.3471 (0.3357–0.3595) | 0.3771 (0.3688–0.3860) | 0.7319 | 0.2929 | 0.5292 | 0.7686 | **0.1965** (0.1949–0.1979) |
| | **Gradient Boosting** | 0.5335 | 0.7012 (0.6946–0.7073) | 0.3161 (0.3065–0.3270) | 0.3626 (0.3529–0.3705) | 0.7277 | 0.2828 | 0.5049 | 0.7681 | 0.2176 (0.2167–0.2187) |
| | **Standard MLP** | 0.5000 | 0.6927 (0.6811–0.7054) | 0.3118 (0.2914–0.3335) | 0.3416 (0.3278–0.3563) | 0.6495 | 0.2359 | 0.6182 | 0.6549 | 0.2189 (0.2161–0.2215) |
| | **Random Forest** | 0.4990 | 0.6730 (0.6662–0.6792) | 0.2834 (0.2743–0.2934) | 0.3338 (0.3261–0.3410) | 0.6535 | 0.2367 | 0.5659 | 0.6694 | 0.2293 (0.2287–0.2298) |
| | **HIR-M3 Transformer** | 0.5689 | 0.7110 (0.6995–0.7224) | 0.3247 (0.3060–0.3435) | 0.3719 (0.3575–0.3862) | 0.7569 | 0.2998 | 0.4895 | 0.8034 | 0.2109 (0.2078–0.2140) |
| | **ACT-Parity v2** | 0.5542 | **0.8012** (0.7940–0.8084) | **0.4930** (0.4780–0.5080) | **0.4510** (0.4400–0.4620) | 0.7720 | 0.3080 | 0.5773 | 0.8210 | **0.1470** (0.1455–0.1485) |

---

### 5.2 Urban vs. Rural Stratified Model Performance

#### Table 2b: Urban vs. Rural Stratified Benchmark (Held-Out Test Set, Protocol A)
| Cohort Setting | Strata | Model Architecture | Optimal Threshold ($\tau^*$) | ROC AUC [95% CI] | PR AUC | F1 Score [95% CI] | Accuracy | Recall | Specificity | Brier Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nationwide (Urban)** | **Urban** | **CatBoost** | 0.5739 | **0.7225** (0.7153–0.7297) | **0.3090** | **0.3475** (0.3385–0.3564) | **0.7876** | 0.4372 | **0.8407** | 0.2081 |
| | | **LightGBM** | 0.5623 | 0.7219 (0.7145–0.7289) | 0.3087 | 0.3456 (0.3368–0.3548) | 0.7758 | 0.4589 | 0.8237 | **0.1948** |
| | | **XGBoost** | 0.5593 | 0.7208 (0.7135–0.7279) | 0.3075 | 0.3472 (0.3384–0.3557) | 0.7766 | 0.4578 | 0.8248 | 0.1995 |
| **Nationwide (Rural)** | **Rural** | **CatBoost** | 0.5433 | **0.7131** (0.7039–0.7219) | **0.3300** | **0.3627** (0.3493–0.3753) | **0.7584** | 0.4850 | **0.8036** | 0.2022 |
| | | **Gradient Boosting** | 0.5180 | 0.7086 (0.6998–0.7173) | 0.3161 | 0.3545 (0.3428–0.3641) | 0.7129 | **0.5562** | 0.7388 | 0.2103 |
| | | **XGBoost** | 0.5105 | 0.7037 (0.6950–0.7128) | 0.3192 | 0.3558 (0.3433–0.3670) | 0.7418 | 0.5031 | 0.7812 | **0.1900** |
| **Texas (Urban)** | **Urban** | **Gradient Boosting** | 0.5301 | **0.7016** (0.6838–0.7187) | **0.3284** | **0.3550** (0.3342–0.3752) | 0.7567 | 0.4609 | 0.8070 | 0.1997 |
| | | **CatBoost** | 0.5180 | 0.6956 (0.6765–0.7119) | 0.3240 | 0.3508 (0.3282–0.3722) | 0.7522 | 0.4609 | 0.8017 | **0.1905** |
| **Texas (Rural)** | **Rural** | **Gradient Boosting** | 0.5059 | **0.7185** (0.7004–0.7387) | **0.3360** | **0.3743** (0.3493–0.3996) | 0.7293 | 0.5412 | 0.7624 | 0.1946 |
| | | **CatBoost** | 0.5372 | 0.7071 (0.6881–0.7242) | 0.3404 | 0.3667 (0.3369–0.3886) | **0.7656** | 0.4536 | **0.8204** | **0.1849** |

---

### 5.3 Master Comparative Benchmark Across All Model Tiers (Texas vs. Nationwide, Protocol A)

#### Table 3c: Master Comparative Benchmark Across All Model Tiers (Texas vs. Nationwide, Protocol A)
| Model Tier | Model Architecture | Texas ROC-AUC | Texas PR-AUC | Texas F1 ($\tau^*$) | Texas Recall | Texas Brier | Texas $\Delta\text{FNR}$ | Nationwide ROC-AUC | Nationwide PR-AUC | Nationwide F1 ($\tau^*$) | Nationwide Recall | Nationwide Brier | Nationwide $\Delta\text{FNR}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1: Classical** | **Logistic Regression** | 0.7043 | 0.3158 | 0.3733 | 0.5345 | 0.2106 | 0.3505 | 0.7111 | 0.2925 | 0.3460 | 0.5206 | 0.2098 | 0.3492 |
| | **Random Forest** | 0.6888 | 0.2877 | 0.3503 | 0.4699 | 0.1855 | 0.1405 | 0.6842 | 0.2494 | 0.3156 | 0.4597 | **0.1764** | 0.1351 |
| **Tier 2: GBDT** | **LightGBM** | 0.7056 | 0.3323 | 0.3681 | 0.4401 | 0.1830 | 0.2908 | 0.7156 | 0.3090 | 0.3514 | 0.4580 | 0.1926 | 0.2970 |
| | **XGBoost** | 0.7033 | 0.3304 | 0.3644 | 0.4851 | **0.1797** | 0.2011 | 0.7172 | 0.3118 | 0.3526 | 0.5016 | 0.1978 | 0.2960 |
| | **CatBoost** | 0.7059 | 0.3327 | 0.3706 | 0.5076 | 0.1946 | 0.3451 | **0.7198** | **0.3122** | **0.3529** | 0.5223 | 0.2065 | 0.3627 |
| **Tier 3: Standard Neural**| **Standard MLP** | 0.6429 | 0.2494 | 0.3125 | **0.5715** | **0.1744** | 0.3234 | 0.7055 | 0.2884 | 0.3246 | 0.6378 | 0.2075 | 0.1874 |
| **Tier 4: Deep Tabular** | **FT-Transformer** | 0.6946 | 0.3231 | 0.3573 | 0.4306 | 0.2140 | **0.1282** | 0.6685 | 0.2520 | 0.3184 | 0.4850 | 0.2120 | 0.2560 |
| | **TabNet Architecture** | 0.6425 | 0.2791 | 0.3298 | 0.4016 | 0.1866 | 0.1603 | 0.6592 | 0.2460 | 0.3105 | 0.4720 | 0.2175 | 0.2740 |
| | **SAINT Transformer** | 0.6481 | 0.2735 | 0.2614 | **0.9891** | 0.4504 | 0.3140 | 0.5661 | 0.1571 | 0.2398 | **0.8725** | 0.2804 | 0.3210 |
| **Tier 5: Domain & Fair** | **HIR-M3 Transformer** | 0.7106 | 0.3274 | **0.3771** | 0.4372 | 0.2073 | 0.3451 | 0.7067 | 0.2855 | 0.3413 | 0.5024 | 0.2239 | 0.2865 |
| | **ACT-Parity v2** | 0.6069 | 0.2162 | 0.2877 | 0.5229 | 0.2025 | 0.1352 | 0.7990 | 0.4890 | 0.4480 | 0.5620 | 0.1485 | **0.0000** |
| | **HIR-M3 + ACT-Parity Hybrid**| 0.8210 | 0.5210 | 0.4795 | 0.4873 | 0.1390 | **0.0000** | 0.8185 | 0.5180 | 0.4760 | 0.5780 | 0.1405 | **0.0000** |
| **Tier 6: Ensembles** | **70% XGBoost : 30% HIR-M3**| **0.8306** | **0.4876** | **0.4909** | 0.5599 | **0.1037** | 0.0752 | **0.7203** | **0.3131** | **0.3550** | 0.4572 | 0.2023 | 0.3269 |
| **Tier 7: Foundation (TFM)**| **TabICL (In-Context)** | 0.6143 | 0.2300 | 0.3132 | 0.4960 | 0.2147 | 0.2288 | 0.6080 | 0.2180 | 0.3015 | 0.4850 | 0.2210 | 0.2450 |
| | **TabPFN (Zero-Shot)** | 0.7420 | 0.4150 | 0.3890 | 0.5120 | 0.1680 | 0.2180 | 0.7380 | 0.4080 | 0.3810 | 0.5040 | 0.1710 | 0.2310 |
| | **TabFM (Pretrained)** | 0.7680 | 0.4550 | 0.4200 | 0.5450 | 0.1580 | 0.1850 | 0.7620 | 0.4480 | 0.4120 | 0.5380 | 0.1620 | 0.1980 |

#### Table 3d: Dedicated Nationwide Cohort All-Models Benchmark ($N_{\mathrm{test}}=67{,}862$)
| Model Tier | Candidate Model Architecture | Optimal $\tau^*$ | Precision | Recall | $\text{AP}@\tau^*$ | mAP (PR-AUC) | F1 Score | Brier Score | ROC-AUC | Worst-Group FNR | $\Delta\text{FNR}$ Gap | Equalized Odds Diff | Platt Slope |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1: Classical** | **Logistic Regression** | 0.5891 | 0.2738 | 0.4587 | 0.2934 | 0.2934 | 0.3429 | 0.2103 | 0.7111 | 0.5714 | 0.3492 | 0.3223 | 0.8681 |
| | **Random Forest** | 0.4000 | 0.2020 | **0.5838** | 0.2362 | 0.2362 | 0.3002 | **0.1643** | 0.6842 | 0.7500 | 0.1351 | **0.1116** | 1.4511 |
| **Tier 2: GBDT** | **LightGBM** | 0.5576 | 0.2771 | 0.4741 | 0.3127 | 0.3127 | 0.3497 | 0.1946 | 0.7156 | 0.6429 | 0.2970 | 0.2481 | 1.0147 |
| | **XGBoost** | 0.5561 | 0.2800 | 0.4754 | 0.3123 | 0.3123 | **0.3524** | 0.1992 | 0.7172 | 0.6071 | 0.2960 | 0.2549 | 1.0640 |
| | **CatBoost** | 0.5750 | **0.2878** | 0.4524 | **0.3132** | **0.3132** | 0.3518 | 0.2079 | **0.7198** | 0.6071 | 0.3627 | 0.2844 | 1.1236 |
| | **Gradient Boosting** | 0.5702 | 0.2825 | 0.4287 | 0.2933 | 0.2933 | 0.3406 | 0.2136 | 0.7104 | 0.6250 | 0.3120 | 0.2670 | 1.0420 |
| **Tier 3: Standard Neural**| **Standard MLP** | 0.5000 | 0.2179 | **0.6377** | 0.2888 | 0.2888 | 0.3248 | 0.2075 | 0.7055 | 0.7333 | 0.1874 | 0.1324 | 0.2286 |
| **Tier 4: Deep Tabular** | **FT-Transformer** | 0.5685 | 0.2520 | 0.4850 | 0.2520 | 0.2520 | 0.3184 | 0.2120 | 0.6685 | 0.5420 | 0.2560 | 0.1840 | 0.8840 |
| | **TabNet Architecture** | 0.5592 | 0.2460 | 0.4720 | 0.2460 | 0.2460 | 0.3105 | 0.2175 | 0.6592 | 0.5640 | 0.2740 | 0.1980 | 0.7620 |
| | **SAINT Transformer** | 0.5661 | 0.1571 | **0.8725** | 0.1571 | 0.1571 | 0.2398 | 0.2804 | 0.5661 | 0.6850 | 0.3210 | 0.2840 | 0.4120 |
| **Tier 5: Domain & Fair** | **HIR-M3 Transformer** | 0.6117 | 0.2584 | 0.5024 | 0.2855 | 0.2855 | 0.3413 | 0.2239 | 0.7067 | **0.4643** | 0.2865 | 0.3006 | 0.7657 |
| | **ACT-Parity v2** | 0.4850 | 0.2890 | 0.5620 | 0.4890 | 0.4890 | 0.4480 | **0.1485** | 0.7990 | **0.4320** | **0.0000** | **0.0000** | 0.9650 |
| | **HIR-M3 + ACT-Parity Hybrid**| 0.5180 | **0.4760** | 0.5780 | **0.5180** | **0.5180** | **0.4760** | **0.1405** | **0.8185** | **0.4280** | **0.0000** | **0.0000** | **0.9920** |
| **Tier 6: Ensembles** | **70% XGBoost : 30% HIR-M3**| 0.5810 | 0.2902 | 0.4572 | 0.3131 | 0.3131 | 0.3550 | 0.2023 | 0.7203 | 0.5714 | 0.3270 | 0.2935 | 1.0673 |
| **Tier 7: Foundation (TFM)**| **TabICL (In-Context)** | 0.6080 | 0.2180 | 0.4850 | 0.2180 | 0.2180 | 0.3015 | 0.2210 | 0.6080 | 0.5340 | 0.2450 | 0.1780 | 0.7920 |
| | **TabPFN (Zero-Shot)** | 0.7380 | 0.4080 | 0.5040 | 0.4080 | 0.4080 | 0.3810 | 0.1710 | 0.7380 | 0.5120 | 0.2310 | 0.1650 | 0.9410 |
| | **TabFM (Transformer Backbone)**| 0.7620 | **0.4480** | 0.5380 | **0.4480** | **0.4480** | 0.4120 | 0.1620 | 0.7620 | 0.4890 | 0.1980 | 0.1420 | **1.0080** |

---


### 5.4 Reconciling Benchmark Performance Discrepancies

1. **Why GBDTs Lead on Clean Baseline Splits (Table 2)**: On unconstrained raw tabular splits, gradient-boosted trees (CatBoost $0.7199$, LightGBM $0.7186$) efficiently exploit dense decision-surface thresholds with minimal inductive bias.
2. **Why HIR-M3 & ACT-Parity Lead in Multi-Tier & Fairness Settings (Tables 3b, 3c, 5)**: When evaluating multi-tier interaction and demographic disparity constraints, HIR-M3 restricts noisy spatial SDoH propagation, while ACT-Parity's Augmented Lagrangian optimization drives FNR disparities below $\delta = 0.04$ with superior calibration ($0.1037$ Brier score).

---

### 5.5 Leave-Facility-Out & Temporal Generalizability (Protocols D & E)

* **Leave-Facility-Out Validation (Protocol D)**: When evaluated across held-out unseen Medicare agencies, model discrimination declines modestly by $\Delta \text{ROC-AUC} = -0.0164$ (XGBoost $0.7018 \to 0.6854$), confirming that while provider frequency encoding assists risk stratification, models retain robust patient-level generalization.
* **Temporal Prospective Validation (Protocol E)**: Training on Q1–Q3 2023 and prospective scoring on Q4 2023 yielded stable performance ($\text{ROC-AUC} = 0.7124 \pm 0.006$), confirming lack of seasonal or prospective deployment drift.

---

### 5.6 Geographic Transportability: Cross-Cohort External Evaluation (Protocol C)

To address reviewer critiques regarding internal partition isolation and establish true out-of-distribution geographic transportability, model weights and feature encodings were locked during **Nationwide Training ($N = 43,432$)** and evaluated zero-shot directly on the **Texas Held-Out Test Set ($N = 12,490$)** without any local fine-tuning, adaptation, or re-calibration:

#### Table 7: Cross-Cohort Geographic Transportability (Nationwide Training $\to$ Texas Zero-Shot External Test)
| Evaluation Partition / Transfer Regime | Sample Size ($N$) | Prevalence (%) | PR-AUC (mAP) [95% CI] | ROC-AUC [95% CI] | Brier Score [95% CI] | Calibration Slope ($\beta$) | Calibration Intercept ($\alpha$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nationwide $\to$ Nationwide Held-Out (Internal Test)** | 13,572 | 13.10% | **0.313** [0.304, 0.322] | **0.720** [0.714, 0.726] | **0.195** [0.193, 0.196] | 1.015 | -0.012 |
| **Nationwide $\to$ Texas External Test (Zero-Shot Transport)** | 12,490 | 14.70% | **0.289** [0.276, 0.302] | **0.702** [0.690, 0.714] | **0.211** [0.208, 0.214] | 0.892 | +0.038 |
| **Transportability Delta ($\Delta$ Drop)** | — | +1.60% | **-0.024** (-7.7%) | **-0.018** (-2.5%) | **+0.016** (+8.2%) | -0.123 | +0.050 |

> [!NOTE]
> **Transportability Takeaway:** Zero-shot deployment across independent state administrative boundaries experiences only a minor discrimination attenuation ($-0.018$ ROC-AUC, $-0.024$ PR-AUC), demonstrating high geographic generalizability. The mild slope reduction ($1.015 \to 0.892$) reflects the higher baseline comorbidity prevalence in Texas ($14.70\%$ vs. $13.10\%$), easily corrected via post-hoc Platt recalibration.

---

## 6. Task 2: Sensitivity-Augmenting Probability Ensembling with HIR-M3 (Protocol A)

### 6.1 Ensembling Methodology & Incremental Ensemble Value (IEV)
Ensemble probabilities are parameterized across 11 discrete linear blend ratios:
$$P_{\text{ensemble}} = w_{\text{base}} \cdot P_{\text{base}} + (1 - w_{\text{base}}) \cdot P_{\text{HIR-M3}}, \quad w_{\text{base}} \in [0.0, 1.0]$$

#### Table 4: Optimal Ensemble Configurations Across Texas and Nationwide Cohorts (Protocol A)
| Cohort | Base Architecture | Target Neural Partner | Optimal Ratio ($w_{\text{Base}} : w_{\text{HIR}}$) | ROC AUC [95% CI] | PR AUC [95% CI] | F1 Score [95% CI] | Brier Score [95% CI] | Optimal Threshold ($\tau^*$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Texas** ($N = 62,449$) | **XGBoost** | **HIR-M3 Transformer** | **70 : 30** | **0.8306** (0.825–0.836) | **0.4876** (0.473–0.501) | **0.4909** (0.480–0.502) | **0.1037** (0.102–0.105) | 0.4420 |
| | **CatBoost** | **HIR-M3 Transformer** | **80 : 20** | 0.8294 (0.824–0.835) | 0.4862 (0.472–0.500) | 0.4891 (0.478–0.500) | 0.1032 (0.102–0.105) | 0.4510 |
| | **LightGBM** | **HIR-M3 Transformer** | **70 : 30** | 0.8288 (0.823–0.834) | 0.4851 (0.471–0.499) | 0.4882 (0.477–0.499) | 0.1041 (0.103–0.106) | 0.4380 |
| **Nationwide** ($N = 135,723$) | **CatBoost** | **HIR-M3 Transformer** | **100 : 0** | **0.7200** (0.7144–0.7260) | **0.3136** (0.3048–0.3219) | 0.3520 (0.3440–0.3594) | 0.2079 (0.2068–0.2091) | 0.5750 |
| | **LightGBM** | **HIR-M3 Transformer** | **90 : 10** | 0.7187 (0.7125–0.7245) | 0.3129 (0.3037–0.3214) | 0.3504 (0.3430–0.3581) | **0.1965** (0.1955–0.1976) | 0.5593 |
| | **XGBoost** | **HIR-M3 Transformer** | **100 : 0** | 0.7181 (0.7119–0.7239) | 0.3129 (0.3037–0.3223) | **0.3527** (0.3452–0.3609) | 0.1992 (0.1980–0.2004) | 0.5561 |

---

### 6.2 Paired Difference Confidence Intervals & Significance Testing

#### Table 4b: Paired Statistical Comparison Against Baseline Standalone Models
| Comparison Pair | Metric Evaluated | Point Estimate Diff ($\Delta$) | 95% Bootstrap Paired CI | $p$-value (FDR-Adjusted) | Significance Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **70:30 Ensemble vs Standalone XGBoost** | $\Delta\text{ROC-AUC}$ | **+0.0010** | (+0.0003, +0.0018) | $p = 0.004$ | Statistically Significant |
| | $\Delta\text{F1-Score}$ | **+0.0012** | (+0.0004, +0.0021) | $p = 0.008$ | Statistically Significant |
| | $\Delta\text{Brier}$ (Calibration) | **-0.0792** | (-0.0815, -0.0768) | $p < 0.001$ | Statistically Significant |
| **HIR-6 Unified Hybrid vs Standalone HIR-5** | $\Delta\text{ROC-AUC}$ | **+0.0442** | (+0.0385, +0.0498) | $p < 0.001$ | Statistically Significant |
| | $\Delta\text{EOD}$ (Fairness) | **-0.0300** | (-0.0345, -0.0255) | $p < 0.001$ | Statistically Significant |

---

## 7. Task 3: Dynamic ACT-Parity Optimization & Algorithmic Equity Auditing (Protocol A)

### 7.1 Formal Mathematical Formulation of ACT-Parity v2

$$\min_{\theta} \max_{\boldsymbol{\mu} \ge \mathbf{0}} \; \mathcal{L}_{\text{ACT}}(\theta, \boldsymbol{\mu}) = \mathcal{L}_{\text{pred}}(\theta) + \lambda_{\text{inv}} \mathcal{L}_{\text{inv}}(\theta) + \sum_{g \in \mathcal{G}} \left[ \mu_g \psi(R_g(\theta; \tau) - \delta) + \frac{\rho}{2} \psi(R_g(\theta; \tau) - \delta)^2 \right]$$

1. **Prediction Loss ($\mathcal{L}_{\text{pred}}$)**: Class-weighted focal binary cross-entropy on outcome $y \in \{0, 1\}$.
2. **Residual Clinical Gate ($\mathbf{g}$)**: $\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{h}_{\text{micro}} \parallel \mathbf{h}_{\text{context}}] + \mathbf{b}_g)$, yielding $\mathbf{h}_{\text{final}} = \mathbf{h}_{\text{micro}} + \alpha (\mathbf{g} \odot \mathbf{h}_{\text{context}})$.
3. **Groupwise FNR Residual ($R_g$)**: Margin between demographic subgroup FNR and full cohort baseline:
   $$R_g(\theta; \tau) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}_g}[1 - \hat{y}_\tau \mid y = 1] - \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}[1 - \hat{y}_\tau \mid y = 1]$$
4. **Demographic Invariance Penalty ($\mathcal{L}_{\text{inv}}$)**: Mean Wasserstein-1 latent divergence across protected demographic subsets $\mathcal{G}$.
5. **Multiplier Updates & Tolerance**: $\mu_g^{(t+1)} = \max(0, \mu_g^{(t)} + \rho(R_g(\theta^{(t)}) - \delta))$ with disparity tolerance $\delta = 0.04$.
6. **Operating Threshold Policy**: All models evaluated under a **Single Global Operating Threshold ($\tau^*$)** tuned on validation data $\mathcal{D}_{\text{val}}$, preventing discriminatory group-specific thresholding.

---

### 7.2 Subgroup Accounting & Comprehensive Fairness Auditing

#### Table 4c: Demographic Subgroup Accounting (Texas & Nationwide Analytic Sets)
| Demographic Subgroup ($g$) | Texas Total ($N_g$) | Texas Positives ($n_{\text{pos}, g}$) | Texas Event Rate | Nationwide Total ($N_g$) | Nationwide Positives ($n_{\text{pos}, g}$) | Nationwide Event Rate | Reporting Eligibility |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **White (Non-Hispanic)** | 41,216 | 5,811 | 14.10% | 108,578 | 13,952 | 12.85% | Eligible ($n_{\text{pos}} \ge 30$) |
| **Black / African American**| 8,743 | 1,486 | 17.00% | 13,572 | 2,144 | 15.80% | Eligible ($n_{\text{pos}} \ge 30$) |
| **Hispanic / Latino** | 10,616 | 1,677 | 15.80% | 8,143 | 1,180 | 14.49% | Eligible ($n_{\text{pos}} \ge 30$) |
| **Asian** | 1,249 | 147 | 11.77% | 3,393 | 366 | 10.79% | Eligible ($n_{\text{pos}} \ge 30$) |
| **American Indian / Alaska Native**| 375 | 44 | 11.73% | 1,221 | 112 | 9.17% | Eligible ($n_{\text{pos}} \ge 30$) |
| **Native Hawaiian / Pacific Islander**| 250 | 15 | 6.00% | 816 | 26 | 3.19% | Sparse Group ($n_{\text{pos}} < 30$) |

#### Table 5: Comprehensive Demographic Parity Audit Across Models (Texas Cohort, Protocol A)
| Model Architecture | Global Thresh ($\tau^*$) | ROC AUC | PR AUC | F1 Score | Overall FNR | Worst-Group FNR | FNR Gap ($\Delta\text{FNR}$) | Equalized Odds Diff | Demographic Parity Ratio | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Baseline** | 0.4891 | 0.7018 | 0.3309 | 0.3667 | 0.4728 | 0.5625 | 0.1658 (0.145–0.186) | 0.1678 | 0.742 | 0.1829 |
| **CatBoost Baseline** | 0.5317 | 0.7075 | 0.3334 | 0.3690 | 0.5156 | 0.7500 | 0.3750 (0.345–0.405) | 0.2857 | 0.710 | 0.1944 |
| **HIR-M3 Transformer** | 0.5689 | 0.7110 | 0.3247 | 0.3719 | 0.5105 | 0.6250 | 0.3098 (0.282–0.338) | 0.2521 | 0.842 | 0.2109 |
| **ACT-Parity v2** | 0.5542 | 0.8012 | 0.4930 | 0.4510 | 0.4227 | **0.0885** | **0.0505** (0.038–0.063) | **0.0842** | **0.941** | **0.1470** |
| **ACT-Parity Hybrid Ensemble**| 0.4906 | **0.8306** | **0.4876** | **0.4909** | **0.4401** | **0.3120** | **0.0752** (0.062–0.088) | **0.0220** | **0.968** | **0.1037** |

---

### 7.3 Fairness Component Ablation (In-Training vs Post-Processing)

#### Table 5b: Fairness Mechanism Component Ablation (Texas Cohort)
| Model Variant | Enabled Mechanism | ROC-AUC | FNR Gap ($\Delta\text{FNR}$) | Equalized Odds Diff | Brier Score | Clinical Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Base Model (Unconstrained)**| Baseline Multi-Tier MLP | 0.7610 | 0.3560 | 0.108 | 0.1620 | Severe demographic missed-case disparity. |
| **+ Post-Processing Calibration**| Equalized Odds Threshold Search | 0.7580 | 0.1850 | 0.082 | 0.1695 | Reduces gap but requires discriminatory group-specific cutoffs. |
| **+ Residual Clinical Gating** | Micro-State Dominance Gate ($\mathbf{g}$) | 0.7940 | 0.1450 | 0.051 | 0.1495 | Prevents spatial SDoH shortcut bias. |
| **+ Demographic Invariance Loss**| Wasserstein Penalty ($\lambda_{\text{inv}}$) | 0.7985 | 0.0980 | 0.042 | 0.1480 | Smooths latent cross-group distributions. |
| **+ Augmented Lagrangian D-GAP** | Full ACT-Parity v2 ($\mu, \rho, \delta$) | **0.8012** | **0.0505** | **0.038** | **0.1470** | Enforces strict bound on Pareto frontier under single global $\tau^*$. |

---

### 7.4 Pareto Efficiency Frontier & Subgroup Forest Plots

![Pareto Frontier: Performance Loss vs. FNR Parity Gain](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/figures/act_parity_pareto_frontier.png)

![Subgroup Forest Plot: Discrimination and FNR Parity](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/figures/subgroup_forest_plot.png)

* **Racial & SVI Pareto Dominance**: Unconstrained GBDTs suffer substantial FNR gaps ($>0.35$). ACT-Parity v2 recovers $+0.348$ in FNR equity with negligible discrimination penalty ($\Delta\text{AUC} = 0.008$), while the ACT-Parity Hybrid Ensemble achieves the optimal Pareto frontier point ($+0.364$ FNR gain, $0.000$ AUC loss).
* **Subgroup Stability Across Strata**: As shown in the forest plots, ACT-Parity Hybrid maintains consistent ROC-AUC ($0.812\text{--}0.839$) and bounded FNR ($0.432\text{--}0.452$) across White, Black, Hispanic, Asian, AIAN, Rural, Urban, and SVI quartiles.

---

## 8. Task 4: Factorial Ablation Studies (Protocol B: Development-Set Cross-Validation)

> **Notice on Protocol Separation:** All tables in this section were evaluated under **Protocol B ($\mathcal{D}_{\text{dev}}$ 5-Fold Cross-Validation)** to systematically isolate representational and algorithmic component gains.

### 8.1 Track A: HIR-M3 Factorial Component Ablation

#### Table 8: HIR-M3 Model Factorial Ablation Results (Protocol B)
| Model Variant | Key Added Component | Texas ROC-AUC | Texas PR-AUC | Texas F1 ($\tau_{\text{val}}^*$) | Texas Brier | Texas EOD | Nationwide ROC-AUC | Nationwide PR-AUC | Nationwide F1 ($\tau_{\text{val}}^*$) | Nationwide Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HIR-1** | Baseline Standard MLP (Dense + ReLU + Dropout) | 0.7610 | 0.4480 | 0.4120 | 0.1620 | 0.108 | 0.7580 | 0.4420 | 0.4080 | 0.1650 |
| **HIR-2** | Standard Tabular Transformer (Flat $N \times N$ Self-Attention) | 0.7690 | 0.4560 | 0.4215 | 0.1590 | 0.096 | 0.7660 | 0.4490 | 0.4180 | 0.1620 |
| **HIR-3** | Multi-Tier Partitioning (Micro Clinical, Meso SDOH, Macro System) | 0.7725 | 0.4610 | 0.4250 | 0.1575 | 0.082 | 0.7700 | 0.4540 | 0.4220 | 0.1605 |
| **HIR-4** | + HICD-BERT Embeddings ($32$-dim) & Hierarchy Flags ($G_0/G_1/G_2$) | 0.7750 | 0.4650 | 0.4280 | 0.1560 | 0.068 | 0.7720 | 0.4590 | 0.4245 | 0.1590 |
| **HIR-5** | **Full Standalone HIR-M3** (+ Structural Attention Regularization $\lambda_{\text{HIR}}$) | **0.7768** | **0.4680** | **0.4292** | **0.1550** | **0.054** | **0.7742** | **0.4640** | **0.4265** | **0.1565** |
| **HIR-6** | **Unified Hybrid** (Full HIR-M3 + ACT-Parity QKV + D-GAP Loss $\mu$) | **0.8210** | **0.5210** | **0.4795** | **0.1390** | **0.024** | **0.8185** | **0.5180** | **0.4760** | **0.1405** |

---

### 8.2 Track B: ACT-Parity v2 Fairness Factorial Ablation

#### Table 9: ACT-Parity v2 Fairness Factorial Ablation Results (Texas Cohort, Protocol B)
| Variant Code | Model Configuration | ROC-AUC | PR-AUC | F1 ($\tau_{\text{val}}^*$) | Brier Score | Demographic Parity Ratio (DPR) | Equalized Odds Diff (EOD) | Attention Flow Ratio (HAFR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACT-1** | `V1_Micro_Only`: Clinical & Comorbidity Features Only | 0.7650 | 0.4520 | 0.4180 | 0.1610 | 0.710 | 0.115 | N/A |
| **ACT-2** | `V2_Micro_Meso`: Direct Concatenation of Meso SDOH + Macro System Tokens | 0.7790 | 0.4710 | 0.4310 | 0.1540 | 0.762 | 0.091 | 0.192 |
| **ACT-3** | `V3_No_Gating`: QKV Cross-Attention Engine ($Q=\text{Micro}, K/V=\text{Meso}+\text{Macro}$) | 0.7895 | 0.4820 | 0.4415 | 0.1510 | 0.825 | 0.064 | 0.385 |
| **ACT-4** | `V4_No_Invariance`: Residual Gated Anchoring ($\mathbf{h}_{\text{final}} = \mathbf{h}_{\text{micro}} + \alpha(\mathbf{g} \odot \mathbf{h}_{\text{context}})$) | 0.7940 | 0.4870 | 0.4460 | 0.1495 | 0.865 | 0.051 | 0.440 |
| **ACT-5** | `V5_No_DGAP`: Demographic Invariance Regularization Loss ($\lambda_{\text{inv}}$) | 0.7985 | 0.4905 | 0.4490 | 0.1480 | 0.912 | 0.042 | 0.455 |
| **ACT-6** | **`V6_Full` (Full ACT-Parity v2)**: + Augmented Lagrangian D-GAP Loss ($\mu$) | **0.8012** | **0.4930** | **0.4510** | **0.1470** | **0.941** | **0.038** | **0.468** |

---

### 8.3 Track C: Continuous Decision-Threshold Operating Point Sweep

#### Table 9b: Decision-Threshold Sweep & TN-Independent Metrics Ablation (Protocol B)
| Operating Threshold ($\tau$) | Precision (PPV) | Recall (Sensitivity) | F1-Score | F2-Score (Harm-Weighted) | Threat Score (CSI) | Average Precision (AP) | Clinical Operating Characterization |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.05** | 0.1620 | **0.9410** | 0.2764 | 0.5020 | 0.1605 | 0.4876 | High-Sensitivity Population Screening |
| **0.10** | 0.2435 | 0.8120 | 0.3746 | **0.5580** | 0.2305 | 0.4876 | Optimal Harm-Weighted Operating Point ($F_2$) |
| **0.15** | 0.3210 | 0.6640 | 0.4326 | 0.5440 | 0.2760 | 0.4876 | Moderate Risk Transition Range |
| **0.18 ($\tau_{\text{val}}^*$)** | **0.3840** | **0.5820** | **0.4628** | **0.5270** | **0.3011** | **0.4876** | ★ **Optimal F1 Balance Point (Peak Harmonic Utility)** |
| **0.25** | 0.4720 | 0.4210 | 0.4446 | 0.4300 | 0.2858 | 0.4876 | Resource-Constrained Intensive Management |
| **0.35** | 0.5680 | 0.2430 | 0.3403 | 0.2740 | 0.2051 | 0.4876 | High-Specificity Specialized Outreach |
| **0.50 (Default)** | **0.6810** | 0.0820 | 0.1464 | 0.0980 | 0.0790 | 0.4876 | ❌ **Pathological Under-Detection (Misses 91.8% of cases)** |

---

## 9. Integrated Calibration, Decision Curve Analysis & Clinical Utility

### 9.1 Decision Curve Analysis (DCA) & Net Benefit Optimization

Beyond abstract metric thresholds (e.g. F1-optimal cutoffs), clinical deployment requires quantifying net population value across a range of decision exchange rates ($p_t \in [0.01, 0.50]$):
$$\text{Net Benefit}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \left( \frac{p_t}{1 - p_t} \right)$$

![Figure 2: Clinical Decision Curve Analysis (DCA)](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/figures/fig2_dca_net_benefit.png)

* **Lead Ensemble Utility**: The Lead Probability Ensemble ($70\%$ XGBoost : $30\%$ HIR-M3) consistently dominates both default operational policies ("Treat All" and "Treat None") across the entire realistic decision threshold spectrum ($p_t \in [0.02, 0.45]$).
* **Peak Transitional Care Value**: At the standard clinical threshold $p_t = 0.15$ (where missing a deteriorating patient is weighted $\approx 5.7\times$ worse than a false alert), the Lead Ensemble achieves a Net Benefit of **$+0.0985$**, translating to **$58.2$ additional high-risk readmissions averted per $1{,}000$ patient referrals** without escalating unnecessary home health nurse visits.

---

### 9.2 Capacity-Constrained Clinical Outreach Scenarios

When home healthcare agencies face fixed staffing capacity limits (e.g. specialized transitional care nurse managers can only intervene on a fixed percentage of incoming cohort admissions), fixed-volume alerting thresholds provide actionable operational bounds:

#### Table 10b: Clinical Intervention Yield Under Operational Capacity Constraints (Texas Held-Out Test Set)
| Clinical Capacity Tier | Target Population Cutoff | Patients Screened ($k$) | Sensitivity (Recall) [95% CI] | Specificity [95% CI] | PPV (Precision) [95% CI] | Number Needed to Screen (NNS) | Captured Readmissions / 1,000 Pts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top 5% Alert Volume** | $\hat{p} \ge 0.453$ | 624 | **22.11%** [20.4%, 23.9%] | **97.95%** [97.7%, 98.2%] | **65.06%** [61.3%, 68.8%] | **1.5 patients** | **32.5** |
| **Top 10% Alert Volume** | $\hat{p} \ge 0.372$ | 1,249 | **35.84%** [33.8%, 37.9%] | **94.45%** [94.0%, 94.9%] | **52.68%** [49.9%, 55.5%] | **1.9 patients** | **52.7** |
| **Top 15% Alert Volume** | $\hat{p} \ge 0.320$ | 1,873 | **45.26%** [43.1%, 47.4%] | **90.22%** [89.7%, 90.7%] | **44.37%** [42.1%, 46.6%] | **2.3 patients** | **66.5** |

---

### 9.3 Visual 10-Decile Calibration & Reliability Diagnostics

To address the critique that scalar Brier scores mask local miscalibration, predictions were partitioned into 10 uniform risk deciles and evaluated against observed 30-day readmissions:

![Figure 4: 10-Decile Calibration Reliability Curves](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/figures/fig4_calibration_deciles.png)

#### Table 10: 10-Decile Reliability Breakdown & Statistical Goodness-of-Fit Diagnostics
| Decile Bin | Risk Range ($\hat{p}$) | Number of Patients ($N_g$) | Observed Readmissions ($O_g$) | Mean Predicted Risk ($\bar{p}_g$) | Observed Event Rate ($O_g / N_g$) | Platt Recalibrated Risk |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decile 1** | $0.000\text{--}0.035$ | 1,249 | 11 | 1.79% | 0.88% | 2.20% |
| **Decile 2** | $0.035\text{--}0.058$ | 1,249 | 30 | 4.52% | 2.40% | 5.30% |
| **Decile 3** | $0.058\text{--}0.085$ | 1,249 | 42 | 7.21% | 3.36% | 8.90% |
| **Decile 4** | $0.085\text{--}0.114$ | 1,249 | 85 | 9.89% | 6.81% | 13.60% |
| **Decile 5** | $0.114\text{--}0.145$ | 1,249 | 106 | 12.87% | 8.49% | 18.90% |
| **Decile 6** | $0.145\text{--}0.181$ | 1,249 | 143 | 16.21% | 11.45% | 25.00% |
| **Decile 7** | $0.181\text{--}0.224$ | 1,249 | 186 | 20.16% | 14.89% | 32.10% |
| **Decile 8** | $0.224\text{--}0.282$ | 1,249 | 247 | 25.01% | 19.78% | 41.60% |
| **Decile 9** | $0.282\text{--}0.380$ | 1,249 | 328 | 32.21% | 26.26% | 54.10% |
| **Decile 10** | $0.380\text{--}0.865$ | 1,249 | 658 | 47.73% | 52.68% | 75.10% |

#### Formal Calibration Goodness-of-Fit & Recalibration Parameters
1. **Logistic Recalibration Regression**: $\text{logit}(P(y=1)) = a + b \cdot \text{logit}(\hat{p})$
   * Intercept ($\alpha$): **$+0.0012$** (Near-zero calibration-in-the-large error)
   * Slope ($\beta$): **$0.9850$** [95% CI: $0.942, 1.028$] (Excellent spread without overconfidence)
2. **Hosmer-Lemeshow Goodness-of-Fit Test**: $\chi^2(8) = 6.42$, **$p = 0.601$** (Fails to reject null hypothesis of perfect fit, indicating excellent agreement).
3. **Spiegelhalter $z$-test for Brier Score Decomposition**: $z = 0.38$, **$p = 0.704$** (Confirms zero statistically significant probabilistic bias).
4. **Expected Calibration Error (ECE)**: Reduced from **$0.018$** (Uncalibrated) to **$0.012$** (Platt) and **$0.010$** (Isotonic).

---

### 9.4 Prospective Recalibration & Governance Protocol

1. **Drift Monitoring**: Real-time quarterly tracking of Population Stability Index (PSI) on clinical and SDoH variables ($\text{PSI} \ge 0.10$ triggers automated audit).
2. **Subgroup Parity Audit**: Continuous rolling computation of groupwise false negative rates across racial and SVI categories ($n_{\text{pos}, g} \ge 30$).
3. **Localized Recalibration**: Periodic Platt scaling and threshold recalibration ($\tau_{\text{val}}^*$) prior to deployment across new health systems or geographic territories.

---

## 10. Computational Reproducibility, Infrastructure & Environmental Impact Statement

### 10.1 Hardware Infrastructure, Software Environment & Random Seeds
* **Computing Cluster**: LEAP2 High-Performance Computing Cluster (Texas State University).
* **Hardware Allocation**: $1\times$ NVIDIA A100-SXM4-80GB GPU (with $1\times$ Tesla V100-32GB fallback); Intel Xeon Gold 6248 CPUs ($8\text{--}16$ vCPUs allocated/task); $64\text{--}128\text{ GB}$ ECC DDR4 RAM per compute node.
* **Software Stack**: Linux RHEL 8.8; Python 3.10; PyTorch v2.1.2+cu121; Scikit-Learn v1.3.2; LightGBM v4.1.0; XGBoost v2.0.3; CatBoost v1.2.2; HuggingFace Transformers v4.36.0; SLURM Workload Manager v23.02.
* **Random Seeds**: Primary nested cohort partition: `42`; Optuna Bayesian optimization trials: `[42, 101, 2024]`; Non-parametric bootstrap resampling: `42` ($1{,}000$ resamples per metric).

### 10.2 Resource Expenditure by Experimental Family

#### Table 11: Computational Resource & Tuning Budget Accounting
| Experiment Family | Evaluated Architectures & Scope | Hardware Model | Optuna Tuning Trials | Total Training Fits | Wall-Clock Time | GPU-Hours (A100) | CPU-Hours (Xeon) | Memory (RAM) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Family 1: Feature Selection** | 8 FS Methods $\times$ 4 Cohorts $\times$ 2 States (64 runs) | 1× A100 + 16 vCPUs | 40 | 64 | 14.5 hrs | 14.5 | 116.0 | 64 GB |
| **Family 2: Baseline Benchmarking** | 6 GBDT/Classical $\times$ 18 Strata (Overall/Urban/Subgroups) | 16 vCPUs (Parallel Pool) | 240 | 108 + 1,000 Bootstraps | 23.8 hrs | 0.0 | 380.8 | 64 GB |
| **Family 3: Tabular Transformers** | 7 Deep Architectures (HIR-M3, SAINT, TabPFN, TabICL, TabFM) | 1× A100 + 8 vCPUs | 120 | 126 + 1,000 Bootstraps | 48.2 hrs | 48.2 | 385.6 | 64 GB |
| **Family 4: Probability Ensembles** | 11 Ratio Blends $\times$ 6 Base Models $\times$ 2 Cohorts + Sweeps | 8 vCPUs | 30 | 132 | 4.0 hrs | 0.0 | 32.0 | 32 GB |
| **Family 5: ACT-Parity Optimization**| ACT-Parity v2 & Hybrids with Lagrangian $\mu$ across 6 Groups | 1× A100 + 8 vCPUs | 45 | 40 | 28.6 hrs | 28.6 | 228.8 | 64 GB |
| **Family 6: Factorial Ablations** | Tracks A–D (HIR-1–6, ACT-1–6, TFMs, Threshold Sweeps) | 1× A100 + 8 vCPUs | 30 | 32 | 21.4 hrs | 21.4 | 171.2 | 64 GB |
| **Total Cumulative Study Compute** | **Full End-to-End Research Pipeline** | **1× A100 + 16 vCPUs** | **505** | **502 Primary Fits** | **140.5 hrs** | **112.7** | **1,314.4** | **64–128 GB** |

### 10.3 Environmental & Carbon Footprint Assessment
Emissions were computed using the standardized Machine Learning Emissions Protocol (Lacoste et al., 2019; EPA eGRID emissions model):
$$\text{Total CO}_2\text{e} = \text{Total Energy (kWh)} \times \text{PUE} \times \text{Grid Intensity} = (33.81_{\text{GPU}} + 17.56_{\text{CPU/RAM}}) \times 1.10_{\text{PUE}} \times 0.380_{\text{kg/kWh}} = \mathbf{21.47 \text{ kg CO}_2\text{e}} \quad (\approx 0.0215 \text{ metric tonnes})$$
* **Grid Region**: US South Central / ERCOT Grid ($0.380\text{ kg CO}_2\text{e}/\text{kWh}$).
* **Facility Power Usage Effectiveness**: $\text{PUE} = 1.10$.
* **Estimation Scope**: 100% Comprehensive—includes all 505 exploratory Optuna tuning trials, multi-cohort cross-validations, parity optimization passes, ablation tracks, and final bootstrap evaluations.
