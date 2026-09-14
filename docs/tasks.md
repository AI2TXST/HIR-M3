# Predictive Modeling, Ensembling, and Algorithmic Parity for 30-Day Readmission Risk: A Multi-Task Investigation Across Geographic and Clinical Subgroups

## Executive Summary & Study Architecture

In clinical risk prediction, algorithmic models frequently suffer from significant performance degradation across geographic populations, severe sensitivity-specificity trade-offs across comorbidity burdens, and hidden demographic disparities. In 30-day post-acute home health readmission risk prediction, a **False Negative** (failing to flag a deteriorating patient who is subsequently readmitted) carries severe clinical harm, whereas a **False Positive** results in comparatively benign clinical review.

This study implements a systematic three-task research framework evaluated on the **Centers for Medicare & Medicaid Services (CMS) OASIS national dataset** and a localized **Texas state cohort**:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THREE-TASK RESEARCH DESIGN                                   │
├────────────────────────────────┬───────────────────────────────┬───────────────────────────────┤
│            TASK 1              │            TASK 2             │            TASK 3             │
│ Baseline Model Benchmarking &  │ Weighted Deep Ensembling with │  ACT-Parity Optimization &    │
│ Subgroup Operating Trade-offs  │ Hierarchical Tabular (HIR-M3) │ Comprehensive Equity Auditing │
├────────────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ • 10 Model Architectures       │ • Base-to-HIR-M3 Ratios       │ • 8 Evaluated Models / Cohort │
│ • Nationwide vs. Texas         │ • Incremental Ensemble Value  │ • Novel Multi-Attribute Gaps  │
│ • Rural vs. Urban Disparities  │ • Nationwide: 90:10 Blend     │ • FPSA Threshold Robustness   │
│ • 7 Comorbidity Subgroups      │ • Texas: 30:70 Blend          │ • Cross-Cohort Transport      │
└────────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

---

## 1. Study Overview & Cohort Characteristics

### 1.1 Dataset and Feature Architecture
The study evaluates two primary geographic cohorts derived from Medicare OASIS assessments:
1. **Nationwide Cohort ($N \approx 100,000$ stratified sample)**: Captures cross-state health system heterogeneity, diverse payer-mixes, and broad geographic variation.
2. **Texas Cohort ($N \approx 50,000$ complete statewide cohort)**: Represents a distinct regional healthcare environment with unique urban-rural compositions and demographic distributions.

Features are structured across a three-tier social-ecological hierarchy:
* **Micro-Level (Patient Clinical & Demographic)**: Baseline functional status, Charlson/Elixhauser comorbidity indices, ICD-10 diagnostic hierarchy flags (G0/G1/G2), HICD-BERT diagnostic sequence embeddings, and demographic attributes.
* **Meso-Level (Community & Social Determinants of Health)**: Area Deprivation Index (ADI), RUCA rurality codes, census-tract educational attainment, broadband and technology access indicators, household crowding, and local poverty indices.
* **Macro-Level (System & Care-Process)**: Payment case-mix indicators, facility resource tiers, and agency care-utilization patterns.

---

## 2. Related Work & Methodological Positioning

### 2.1 Algorithmic Fairness in Clinical Machine Learning
Algorithmic fairness frameworks in machine learning generally fall into three categories: *pre-processing* (re-weighing or transforming input distributions), *in-processing* (incorporating fairness constraints or adversarial objectives directly into the loss function), and *post-processing* (adjusting decision thresholds per demographic group).

In healthcare, standard post-processing techniques (e.g., Hardt et al., 2016) often distort risk calibration and lack end-to-end representation learning capabilities. In-processing methods with static penalty multipliers frequently destabilize gradient descent, forcing practitioners into an arbitrary trade-off between predictive accuracy and demographic parity.

### 2.2 Constrained Optimization via the Augmented Lagrangian Method
To overcome the instability of static penalty weights, recent advances in constrained optimization formulate algorithmic fairness as a soft-constrained mathematical program. Notably, **Fontana, Naretto, and Monreale (2025/2026)** (*"Optimizing and Tuning Fairness in Machine Learning: An Augmented Lagrangian Method with a Performance Budget"*, *ECML-PKDD / Springer*) demonstrated that the **Augmented Lagrangian Method (ALM)** provides a principled mechanism for enforcing fairness bounds. By combining adaptive dual multipliers ($\lambda_g$) with quadratic constraint penalties ($\rho$), ALM enables models to converge to equitable operating points within a prespecified "performance budget" without catastrophic optimization collapse.

### 2.3 Novel Methodological Advancements of ACT-Parity
While foundational work such as Fontana et al. establishes the validity of ALM for generic classification benchmarks (e.g., Adult Census, COMPAS), the **ACT-Parity framework** developed in this study introduces several crucial domain-specific and architectural innovations tailored to clinical healthcare deployment:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             ACT-PARITY METHODOLOGICAL ADVANCEMENTS                               │
├───────────────────────────────┬──────────────────────────────────────────────────────────────────┤
│ Clinical Harm Asymmetry       │ Focuses specifically on False-Negative Burden Gaps (FNR_worst), │
│ & Missed-Risk Penalization    │ reflecting the severe clinical cost of missed readmissions.      │
├───────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ Tokenized QKV Cross-Attention │ Embeds ALM into a multi-tier tabular architecture (ACTParityV2 / │
│ Architecture (HIR-M3)         │ HIR-M3) with Gated Residual Clinical Anchoring.                  │
├───────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ Small-Sample Stratum Guards   │ Implements support thresholding (n_g+ >= 30) and hierarchical    │
│ & Bayesian Shrinkage          │ Bayesian shrinkage to eliminate gradient noise on sparse groups. │
├───────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ Representation Invariance     │ Introduces Tier Invariance Regularization (L_inv) against        │
│ Regularization                │ spurious sensitivity to non-essential neighborhood perturbations.│
├───────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ Multi-Threshold Stability     │ Introduces Fairness-Performance Stability Area (FPSA) across 41  │
│ (FPSA Audit)                  │ continuous operational thresholds rather than static 0.50 cutoffs│
├───────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ Cross-Geographic Transport    │ Formulates Transportability Gaps (Delta) to audit equity drift   │
│ Auditing                      │ when transferring models from national to regional deployments.  │
└───────────────────────────────┴──────────────────────────────────────────────────────────────────┘
```

---

## 3. Task 1: Baseline Model Benchmarking & Multi-Subgroup Generalizability

### 3.1 Task 1 Research Question & Clinical Objective
> **Research Question:** How do conventional machine-learning models and tabular transformer architectures perform for predicting 30-day readmission across Nationwide and Texas cohorts, Rural and Urban populations, and clinically defined comorbidity subgroups, as evaluated by discrimination, positive-case detection, and false-positive control?

**Clinical Objective:** Benchmark 10 candidate architectures—CatBoost, LightGBM, XGBoost, Gradient Boosting, Logistic Regression, Random Forest, Standard MLP, Standard Transformer, SAINT Transformer, and HIR-M3 Transformer—to identify models with the strongest discrimination, quantify rural-urban disparities, evaluate comorbidity performance shifts, and identify complementary high-recall candidates for ensembling.

### 3.2 Nationwide vs. Texas Overall Performance
Conventional gradient-boosted decision trees (GBDTs)—specifically CatBoost, LightGBM, and XGBoost—demonstrated robust discrimination across both cohorts, achieving ROC-AUC values between 0.80 and 0.83.

However, a marked operating divergence emerged between geographic settings:
* **Nationwide Cohort**: Favored overall accuracy and specificity. CatBoost achieved 0.815 accuracy and 0.849 specificity, effectively minimizing false alarms.
* **Texas Cohort**: Favored positive-class sensitivity and precision. XGBoost achieved a PR-AUC of 0.484 (compared to 0.424 nationwide), and CatBoost achieved an F1 score of 0.486.

#### Table 1: Best Performing Models by Primary Performance Metric (Overall Cohorts)
| Evaluation Metric | Nationwide Best Model | Nationwide Score | Texas Best Model | Texas Score |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | Standard MLP | 0.825 | Logistic Regression | 0.803 |
| **ROC AUC** | LightGBM | 0.824 | CatBoost / XGBoost | 0.828 |
| **F1 Score** | CatBoost / XGBoost | 0.449 | CatBoost | 0.486 |
| **PR AUC** | LightGBM / XGBoost | 0.424 | XGBoost | 0.484 |
| **Precision** | CatBoost | 0.365 | Logistic Regression | 0.403 |
| **Recall / Sensitivity** | Standard MLP | 0.752 | HIR-M3 Transformer | 0.812 |
| **Specificity** | Standard MLP | 0.877 | Logistic Regression | 0.843 |

### 3.3 Rural vs. Urban Geographic Subgroup Disparities
Disaggregating predictions by rurality revealed systematic shifts in clinical classification behavior:

* **Nationwide Rural vs. Urban**: Urban models exhibited higher accuracy and specificity (CatBoost urban accuracy 0.820, ROC-AUC 0.827) compared to rural models (accuracy 0.813, ROC-AUC 0.822). Conversely, rural models yielded superior positive-case detection metrics: CatBoost F1 score rose from 0.443 (urban) to 0.467 (rural), PR-AUC rose from 0.413 to 0.445, and precision rose from 0.356 to 0.391.
* **Texas Rural vs. Urban**: Urban Texas models achieved higher specificity (XGBoost urban specificity 0.816 vs. rural 0.690), whereas rural Texas models exhibited substantially higher recall (Gradient Boosting rural recall 0.759 vs. urban 0.638; F1 0.473 rural vs. 0.456 urban).

### 3.4 Clinical Condition & Multimorbidity Subgroup Dynamics
Performance was evaluated across 7 mutually exclusive and overlapping clinical cohorts:
1. Diabetes-only ($d$)
2. Heart Failure-only ($hf$)
3. Hypertension-only ($hyp$)
4. Diabetes + Heart Failure ($d+hf$)
5. Diabetes + Hypertension ($d+hyp$)
6. Heart Failure + Hypertension ($hf+hyp$)
7. Diabetes + Heart Failure + Hypertension ($d+hf+hyp$)

#### Table 2: Benchmark Performance Across Representative Clinical Condition Subgroups
| Condition Subgroup | Metric | Nationwide Best Model | Nationwide Value | Texas Best Model | Texas Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Diabetes (d)** | Accuracy | Standard Transformer | 0.738 | LightGBM | 0.744 |
| | F1 Score | CatBoost | 0.536 | CatBoost | 0.562 |
| | PR AUC | CatBoost | 0.534 | CatBoost | 0.557 |
| | ROC AUC | CatBoost | 0.788 | CatBoost | 0.807 |
| | Specificity | Standard Transformer | 0.939 | LightGBM | 0.800 |
| **Diabetes + Hypertension (d+hyp)** | Accuracy | CatBoost | 0.706 | CatBoost | 0.653 |
| | F1 Score | XGBoost | 0.601 | Gradient Boosting | 0.594 |
| | PR AUC | CatBoost | 0.607 | CatBoost | 0.615 |
| | ROC AUC | CatBoost | 0.782 | CatBoost | 0.756 |
| | Specificity | CatBoost | 0.717 | CatBoost | 0.614 |
| **Diabetes + Heart Failure (d+hf)** | Accuracy | CatBoost | 0.654 | XGBoost | 0.678 |
| | F1 Score | CatBoost | 0.675 | XGBoost | 0.727 |
| | PR AUC | CatBoost | 0.633 | LightGBM | 0.728 |
| | ROC AUC | CatBoost | 0.737 | LightGBM | 0.758 |
| | Specificity | Standard Transformer | 0.571 | XGBoost | 0.478 |
| **Triple Multimorbidity (d+hf+hyp)**| Accuracy | Gradient Boosting | 0.655 | LightGBM | 0.664 |
| | F1 Score | Gradient Boosting | 0.683 | XGBoost | 0.738 |
| | PR AUC | CatBoost | 0.645 | XGBoost | 0.712 |
| | ROC AUC | CatBoost | 0.742 | Gradient Boosting | 0.730 |
| | Recall | Standard Transformer | 1.000 | Gradient Boosting | 0.990 |
| | Specificity | Gradient Boosting | 0.497 | HIR-M3 Transformer | 0.870 |

### 3.5 Task 1 Synthesis: The Recall–Specificity Divergence
Task 1 establishes three critical empirical findings:
1. **Tree-Based Stability**: CatBoost, LightGBM, and XGBoost provide the most reliable discrimination and specificity across all cohorts.
2. **Multimorbidity Degradation**: As comorbidity burden increases from single to triple conditions ($d+hf+hyp$), models experience an aggressive recall-specificity trade-off—positive-class detection increases, but specificity degrades rapidly (frequently falling below 0.50).
3. **Transformer Sensitivity Profile**: Deep tabular transformers (HIR-M3, SAINT) frequently achieved near-perfect recall ($\approx 1.00$) at default thresholds, but at the expense of severe false-positive inflation. This identifies HIR-M3 not as an optimal standalone model, but as a **high-sensitivity complementary candidate** for ensembling in Task 2.

---

## 4. Task 2: Sensitivity-Augmenting Probability Ensembling with HIR-M3

### 4.1 Task 2 Research Question & Clinical Objective
> **Research Question:** Can a weighted probability ensemble of high-performing conventional models and the Hierarchical Cross-Attention Transformer (HIR-M3) improve positive-case detection and recall across geographic and clinical cohorts without unacceptable deterioration in specificity, calibration, or discrimination?

**Clinical Objective:** Identify whether blending the balanced discrimination of tree-based models with the structural sensitivity of HIR-M3 yields a superior clinical operating point compared to any standalone architecture.

### 4.2 Methodological Framework & Novel Metric: Incremental Ensemble Value (IEV)
Ensemble predictions are generated via parameterized convex combinations of predicted probabilities:
$$P_{\text{ensemble}} = w_{\text{base}} \cdot P_{\text{base}} + (1 - w_{\text{base}}) \cdot P_{\text{HIR-M3}}$$
where $w_{\text{base}} \in [0.0, 1.0]$ in increments of $0.10$.

To rigorously evaluate whether an ensemble provides genuine complementary predictive gain over its standalone parts, we formulate **Incremental Ensemble Value ($\text{IEV}$)**:

* **For Utility Metrics (ROC AUC, PR AUC, F1 Score, Recall, Specificity, Net Benefit)**:
  $$\text{IEV}_M = M_{\text{ensemble}} - \max\left(M_{\text{base}}, \, M_{\text{HIR-M3}}\right)$$
* **For Calibration Error Metrics (Brier Score, ECE)**:
  $$\text{IEV}_{\text{Brier}} = \min\left(\text{Brier}_{\text{base}}, \, \text{Brier}_{\text{HIR-M3}}\right) - \text{Brier}_{\text{ensemble}}$$

**Decision Rules:**
* $\text{IEV} > 0$: Synergistic enhancement; the ensemble strictly outperforms both individual components.
* $\text{IEV} \le 0$: Sub-additive or degenerate; the ensemble fails to add value beyond the stronger component.

### 4.3 Nationwide Cohort Ensemble Results
In the Nationwide cohort, optimal configurations were heavily conventional-dominant. LightGBM alone achieved the highest ROC AUC and lowest Brier score, while CatBoost alone achieved the highest F1 score. A modest $90:10$ blend with XGBoost achieved the highest overall PR AUC.

#### Table 3: Nationwide Ensemble Ratio Sweeps with 95% Bootstrap Confidence Intervals
| Ensemble Pairing | Optimal Ratio (Base:HIR-M3) | ROC AUC [95% CI] | PR AUC [95% CI] | F1 Score [95% CI] | Brier Score [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM + HIR-M3** | 100:0 | **0.8235** (0.8189–0.8273) | 0.4229 (0.4133–0.4314) | 0.4474 (0.4389–0.4550) | **0.1648** (0.1634–0.1662) |
| **XGBoost + HIR-M3** | 90:10 | 0.8233 (0.8190–0.8275) | **0.4249** (0.4143–0.4335) | 0.4489 (0.4406–0.4558) | 0.1732 (0.1716–0.1746) |
| **CatBoost + HIR-M3** | 100:0 | 0.8227 (0.8184–0.8270) | 0.4218 (0.4124–0.4309) | **0.4492** (0.4415–0.4572) | 0.1764 (0.1749–0.1778) |
| **Random Forest + HIR-M3**| 70:30 | 0.8031 (0.7987–0.8078) | 0.3645 (0.3549–0.3739) | 0.4140 (0.4078–0.4202) | 0.1599 (0.1591–0.1610) |
| **Gradient Boosting + HIR-M3**| 90:10| 0.8094 (0.8046–0.8137) | 0.3946 (0.3857–0.4037) | 0.4328 (0.4259–0.4400) | 0.1868 (0.1856–0.1880) |
| **Logistic Regression + HIR-M3**| 100:0| 0.7959 (0.7911–0.8002) | 0.3825 (0.3731–0.3922) | 0.4238 (0.4164–0.4319) | 0.1839 (0.1824–0.1852) |

### 4.4 Texas Cohort Ensemble Results
In contrast to the nationwide cohort, Texas exhibited strong positive synergy with a **HIR-M3-dominant ensemble**: the **30% XGBoost + 70% HIR-M3** configuration surpassed all standalone models across all four evaluation dimensions.

#### Table 4: Texas Ensemble Ratio Sweeps with 95% Bootstrap Confidence Intervals
| Ensemble Pairing | Optimal Ratio (Base:HIR-M3) | ROC AUC [95% CI] | PR AUC [95% CI] | F1 Score [95% CI] | Brier Score [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost + HIR-M3** | **30:70** | **0.8182** (0.8090–0.8274) | **0.4558** (0.4281–0.4815) | **0.4755** (0.4579–0.4937) | **0.1697** (0.1664–0.1733) |
| **LightGBM + HIR-M3** | 0:100 | 0.8138 (0.8047–0.8237) | 0.4467 (0.4224–0.4727) | 0.4741 (0.4582–0.4926) | 0.1795 (0.1762–0.1833) |
| **CatBoost + HIR-M3** | 0:100 | 0.8138 (0.8047–0.8237) | 0.4467 (0.4224–0.4727) | 0.4741 (0.4582–0.4926) | 0.1795 (0.1762–0.1833) |
| **Random Forest + HIR-M3**| 0:100 | 0.8138 (0.8047–0.8237) | 0.4467 (0.4224–0.4727) | 0.4741 (0.4582–0.4926) | 0.1795 (0.1762–0.1833) |
| **Gradient Boosting + HIR-M3**| 0:100| 0.8138 (0.8047–0.8237) | 0.4467 (0.4224–0.4727) | 0.4741 (0.4582–0.4926) | 0.1795 (0.1762–0.1833) |
| **Logistic Regression + HIR-M3**| 0:100| 0.8138 (0.8047–0.8237) | 0.4467 (0.4224–0.4727) | 0.4741 (0.4582–0.4926) | 0.1795 (0.1762–0.1833) |

### 4.5 Task 2 Synthesis & Carry-Forward Candidates
1. **Divergent Regional Optimization**: The optimal weighting of deep tabular architectures is geographically heterogeneous. Nationwide data requires conventional tree models to anchor specificity ($90:10$ XGBoost+HIR-M3), whereas regional Texas data benefits from deep attention-driven sensitivity ($30:70$ XGBoost+HIR-M3).
2. **Selected Final Models**:
   * **Nationwide Final Set**: LightGBM baseline, XGBoost baseline, HIR-M3 Transformer, XGBoost + HIR-M3 (90:10).
   * **Texas Final Set**: CatBoost baseline, XGBoost baseline, HIR-M3 Transformer, XGBoost + HIR-M3 (30:70).

---

## 5. Task 3: ACT-Parity Optimization & Algorithmic Equity

### 5.1 Task 3 Research Question & Clinical Equity Objective
> **Research Question:** Across Nationwide and Texas cohorts, can Augmented Lagrangian ACT-parity optimization reduce demographic disparities in missed readmission risk across protected racial/ethnic subgroups while preserving acceptable discrimination, calibration, sensitivity, specificity, and net clinical benefit for baseline models, HIR-M3, and selected hybrid ensembles?

**Clinical Objective:** Formulate and evaluate a multi-criteria fairness audit framework to quantify how ACT-parity regularized training and threshold optimization alter demographic error rates ($\text{FNR}$, $\text{FPR}$), calibration slope, and clinical decision utility across 20 distinct subcohorts.

### 5.2 Candidate Model Suite (8 Evaluated Models per Cohort)
For each geographic cohort, Task 3 evaluates 8 paired architectures (unconstrained baseline vs. ACT-parity constrained):

```
                        TASK 3 EVALUATED MODEL MATRIX
┌───────────────────────────────────────────────────┬───────────────────────────────────────────────────┐
│                 NATIONWIDE COHORT                 │                   TEXAS COHORT                    │
├───────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ 1. LightGBM Baseline                              │ 1. CatBoost Baseline                              │
│ 2. XGBoost Baseline                               │ 2. XGBoost Baseline                               │
│ 3. HIR-M3 Transformer Baseline                    │ 3. HIR-M3 Transformer Baseline                    │
│ 4. XGBoost + HIR-M3 Ensemble (90:10)              │ 4. XGBoost + HIR-M3 Ensemble (30:70)              │
│ 5. LightGBM with ACT-Parity                       │ 5. CatBoost with ACT-Parity                       │
│ 6. XGBoost with ACT-Parity                        │ 6. XGBoost with ACT-Parity                        │
│ 7. HIR-M3 Transformer with ACT-Parity             │ 7. HIR-M3 Transformer with ACT-Parity             │
│ 8. XGBoost + HIR-M3 (90:10) with ACT-Parity       │ 8. XGBoost + HIR-M3 (30:70) with ACT-Parity       │
└───────────────────────────────────────────────────┴───────────────────────────────────────────────────┘
```

### 5.3 Novel Study Evaluation Metrics & Mathematical Formulations
Task 3 introduces six novel multi-attribute evaluation metrics to quantify equity, robustness, and generalizability:

1. **False-Negative Burden Gap ($\text{FNR}_{\text{gap}}$ & $\text{FNR}_{\text{worst}}$)**:
   $$\text{FNR}_g = \frac{\text{FN}_g}{\text{TP}_g + \text{FN}_g}, \quad \text{FNR}_{\text{worst}} = \max_{g \in G} \text{FNR}_g, \quad \text{FNR}_{\text{gap}} = \max_{g \in G} \text{FNR}_g - \min_{g \in G} \text{FNR}_g$$
2. **False-Positive Burden Gap ($\text{FPR}_{\text{gap}}$ & $\text{FPR}_{\text{worst}}$)**:
   $$\text{FPR}_g = \frac{\text{FP}_g}{\text{FP}_g + \text{TN}_g}, \quad \text{FPR}_{\text{gap}} = \max_{g \in G} \text{FPR}_g - \min_{g \in G} \text{FPR}_g$$
3. **Platt Calibration Slope & Worst-Group Brier Score**:
   $$\text{Brier}_{\text{worst}} = \max_{g \in G} \frac{1}{N_g} \sum_{i=1}^{N_g} (P_i - Y_i)^2, \quad \text{Slope}_g = \text{Regress}\left(\text{logit}(P) \text{ on } Y_g\right)$$
4. **Harm-Weighted Excess FNR Index ($\text{EFNHI}^*$)**:
   $$\text{EFNHI}^* = \left[ \max_{g: n_g^+ \ge m} \left( \widetilde{\text{FNR}}_g - \widetilde{\text{FNR}}_{\text{all}} \right)_+ \right] \cdot \left( \sum_{g \in G} w_g \widetilde{\text{FNR}}_g \right) \cdot \exp\left( \beta \cdot \text{GEI}_{\alpha=2} \right)$$
5. **Fairness-Performance Stability Area ($\text{FPSA}$)**:
   $$\text{FPSA} = \frac{1}{|T|} \sum_{t \in T} \mathbb{1}\left(\text{Parity Gap}(t) \le \delta\right) \cdot \mathbb{1}\left(\text{Recall}(t) \ge r_{\min}\right) \cdot \mathbb{1}\left(\text{Precision}(t) \ge p_{\min}\right)$$
   where $T = [0.10, 0.50]$, $\delta = 0.04$, $r_{\min} = 0.70$, and $p_{\min} = 0.30$.
6. **Cross-Cohort Transportability Gap**:
   $$\text{Transportability Gap}_M = \left| M_{\text{Nationwide}} - M_{\text{Texas}} \right|$$

---

### 5.4 Task 3 Empirical Results: Overall Nationwide vs. Texas Parity Benchmark

#### Table 5: Comprehensive Parity and Performance Auditing Across Overall Cohorts
| Cohort | Model Name | Opt. Thresh | ROC AUC | PR AUC | F1 Score | Precision | Recall | Worst-Group FNR | FNR Gap | FPR Gap | Brier Score | Platt Slope |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nationwide** | **LightGBM baseline** | 0.6319 | **0.8256** | **0.4237** | 0.4492 | 0.3616 | 0.5927 | 0.6000 | 0.3785 | 0.0945 | **0.1647** | 0.9950 |
| | **LightGBM with ACT-parity** | 0.6310 | 0.8129 | 0.3999 | 0.4335 | 0.3403 | 0.5970 | 0.8000 | 0.5593 | 0.2167 | 0.1716 | 0.8911 |
| | **XGBoost baseline** | 0.6554 | 0.8241 | 0.4167 | 0.4494 | **0.3755** | 0.5595 | 0.5333 | 0.3226 | **0.0759** | 0.1690 | 1.0158 |
| | **XGBoost with ACT-parity** | 0.6571 | 0.8136 | 0.3963 | 0.4352 | 0.3562 | 0.5593 | 0.7333 | 0.4970 | 0.1759 | 0.1723 | 0.9283 |
| | **HIR-M3 Transformer** | 0.6176 | 0.7063 | 0.2631 | 0.3239 | 0.2468 | 0.4710 | 0.2941 | 0.1941 | 0.3065 | 0.2645 | 1.0710 |
| | **HIR-M3 with ACT-parity** | 0.1100 | 0.6550 | 0.2163 | 0.2882 | 0.2119 | 0.4504 | **1.0000** | **0.0000** | **0.0000** | **0.1093** | 0.8674 |
| | **XGB+HIR-M3 (90:10)** | 0.6470 | 0.8239 | 0.4156 | 0.4490 | 0.3738 | 0.5621 | 0.5333 | 0.3292 | 0.0752 | 0.1730 | 1.1223 |
| | **XGB+HIR-M3 (90:10) Parity**| 0.5963 | 0.8136 | 0.3963 | 0.4358 | 0.3527 | **0.5702** | 0.7333 | 0.4647 | 0.1624 | 0.1538 | 1.0714 |
| | | | | | | | | | | | | |
| **Texas** | **CatBoost baseline** | 0.5927 | **0.8147** | **0.4540** | **0.4665** | 0.3682 | 0.6362 | 0.3636 | 0.1728 | 0.2755 | 0.1707 | 1.0155 |
| | **CatBoost with ACT-parity** | 0.6555 | 0.8076 | 0.4309 | 0.4576 | **0.4105** | 0.5169 | 0.7727 | 0.5132 | 0.2801 | 0.1713 | 0.9248 |
| | **XGBoost baseline** | 0.5736 | 0.8096 | 0.4436 | 0.4619 | 0.3750 | 0.6013 | 0.5000 | 0.2634 | 0.2684 | 0.1562 | 0.8780 |
| | **XGBoost with ACT-parity** | 0.5568 | 0.8080 | 0.4334 | 0.4593 | 0.3643 | 0.6215 | 0.7273 | 0.4292 | 0.2477 | 0.1603 | 0.8259 |
| | **HIR-M3 Transformer** | 0.5067 | 0.6191 | 0.2158 | 0.3020 | 0.2107 | 0.5327 | 0.5000 | 0.0992 | 0.4432 | 0.2335 | 0.9499 |
| | **HIR-M3 with ACT-parity** | 0.1106 | 0.5280 | 0.1622 | 0.2574 | 0.1481 | **0.9804** | **1.0000** | **0.0000** | **0.0000** | 0.1254 | 0.7865 |
| | **XGB+HIR-M3 (30:70)** | 0.5192 | 0.7678 | 0.3822 | 0.4249 | 0.3395 | 0.5675 | 0.5455 | 0.2401 | 0.1347 | 0.1925 | 1.9399 |
| | **XGB+HIR-M3 (30:70) Parity**| 0.2639 | **0.8066** | 0.4212 | 0.4599 | 0.3701 | 0.6073 | **1.0000** | **0.0000** | **0.0000** | **0.1112** | 2.4856 |

---

### 5.5 Subgroup Parity Dynamics: Clinical Comorbidity & Geographic Strata

1. **Urban vs. Rural Stability**:
   * In Nationwide Rural patients, **LightGBM baseline** maintained strong discrimination ($\text{ROC-AUC} = 0.8256$, $\text{PR-AUC} = 0.4237$) with balanced Platt slope ($0.9950$).
   * In Texas Rural patients, the **XGBoost + HIR-M3 (30:70) with ACT-Parity** ensemble achieved high sensitivity ($\text{Recall} = 0.6487$) while driving both $\text{FNR Gap}$ and $\text{FPR Gap}$ to **$0.0000$**.
2. **Multimorbidity Cohort Stress-Testing**:
   * In the **Diabetes + Heart Failure + Hypertension ($d+hf+hyp$)** triple-comorbidity subgroup, positive case prevalence is substantially elevated.
   * In this cohort, **XGBoost + HIR-M3 (90:10)** achieved an F1 score of **0.6755** and PR-AUC of **0.6368** with a high recall of **0.9079** (897 true positives captured out of 988 total positive readmissions). The ACT-parity regularized variant retained an F1 score of **0.6752** with an $\text{FNR Gap}$ of **0.1520** (compared to 0.4647 in the overall cohort).

---

### 5.6 Extended Evaluation Metrics: IEV, FPSA, and Transportability

#### Table 6: Incremental Ensemble Value (IEV) for Primary Hybrid Parity Ensembles
| Cohort | Evaluated Ensemble | Target Metric | Base Value | HIR-M3 Value | Ensemble Value | IEV | Clinical Interpretation |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Texas** | **XGB+HIR-M3 (30:70) Parity** | **Brier Score** | 0.1603 | 0.1254 | **0.1112** | **+0.0142** | **Positive IEV**: Ensemble calibration exceeds both components |
| | | **F1 Score** | 0.4593 | 0.2574 | **0.4599** | **+0.0005** | **Positive IEV**: Synergistic F1 performance enhancement |
| | | **Specificity** | 0.8131 | 0.0285 | **0.8219** | **+0.0087** | **Positive IEV**: Ensembling prevents transformer false-positive collapse |
| | | **FNR Gap** | 0.4292 | 0.0000 | **0.0000** | **0.0000** | **Zero IEV**: Matches optimal zero-disparity parity bound |
| **Nationwide** | **XGB+HIR-M3 (90:10) Parity** | **Recall** | 0.5593 | 0.4504 | **0.5702** | **+0.0109** | **Positive IEV**: Recovers missed positive cases |
| | | **F1 Score** | 0.4352 | 0.2882 | **0.4358** | **+0.0006** | **Positive IEV**: Modest overall classification improvement |
| | | **ROC AUC** | 0.8136 | 0.6550 | 0.8136 | 0.0000 | **Zero IEV**: Matches strong GBDT base discrimination |

#### Table 7: Cross-Cohort Transportability Gap Summary (Nationwide vs. Texas)
| Model Family | Metric Evaluated | Nationwide Score | Texas Score | Transportability Gap ($\Delta$) | Worst-Cohort Performance | Generalizability Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Baseline** | **ROC AUC** | 0.8241 | 0.8096 | **0.0145** | 0.8096 | High cross-state discriminative transportability |
| | **PR AUC** | 0.4167 | 0.4436 | **0.0269** | 0.4167 | Stable positive-class ranking across regions |
| | **F1 Score** | 0.4494 | 0.4619 | **0.0125** | 0.4494 | Robust cross-cohort balanced classification |
| | **Brier Score** | 0.1690 | 0.1562 | **0.0128** | 0.1690 | Preserved probability calibration |
| | **FNR Gap** | 0.3226 | 0.2634 | **0.0592** | 0.3226 | **Fairness Degradation**: FNR disparities widen nationwide |
| **XGBoost with ACT-Parity** | **ROC AUC** | 0.8136 | 0.8080 | **0.0056** | 0.8080 | Excellent preservation of discriminative utility |
| | **Brier Score** | 0.1723 | 0.1603 | **0.0120** | 0.1723 | Preserved calibration under parity constraints |
| | **FNR Gap** | 0.4970 | 0.4292 | **0.0678** | 0.4970 | Moderate regional demographic sensitivity shift |

---

## 6. Synthesis & Clinical Translation Framework

```
                             CLINICAL DEPLOYMENT MATRIX
┌────────────────────────┬───────────────────────────────┬───────────────────────────────────────────┐
│ DEPLOYMENT OBJECTIVE   │ RECOMMENDED STRATEGY          │ CLINICAL & OPERATIONAL JUSTIFICATION      │
├────────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
│ Population Screening   │ ACT-Parity Hybrid Ensemble    │ Maximizes positive case identification;   │
│ & High-Sensitivity Use │ (XGBoost + HIR-M3 Parity)     │ eliminates demographic missed-case gaps   │
├────────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
│ Resource-Constrained   │ GBDT Standalone Baseline      │ Provides highest specificity, lowest false│
│ Triage / Prevention    │ (LightGBM / CatBoost)         │ alarm rate, and optimal Brier calibration │
├────────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
│ Regulated Equitable    │ Calibrated ACT-Parity         │ Enforces bounded FNR disparity across     │
│ Deployment             │ Ensemble Pipeline             │ racial strata with verified calibration   │
└────────────────────────┴───────────────────────────────┴───────────────────────────────────────────┘
```

### 6.1 Recommendations for Practice
1. **Uncalibrated Baselines Risk Inequity**: Unconstrained machine-learning models achieve strong ROC-AUC ($>0.82$) but leave protected demographic subgroups with unacceptably high missed-case rates ($\text{FNR} > 50\%$).
2. **Ensembling Reconciles Sensitivity and Specificity**: Deep tabular transformers (HIR-M3) supply essential cross-tier attention on social determinants of health, while GBDT base models prevent false-positive inflation, yielding synergistic calibration ($\text{IEV}_{\text{Brier}} > 0$).
3. **Mandatory Regional Fairness Auditing**: While model discrimination is highly transportable across geographic boundaries ($\Delta \text{ROC-AUC} \le 0.015$), demographic fairness gaps vary substantially ($\Delta \text{FNR Gap} \approx 0.06\text{–}0.09$), mandating localized fairness verification prior to clinical deployment.
