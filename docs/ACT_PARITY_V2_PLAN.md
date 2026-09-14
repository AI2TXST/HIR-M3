# ACT-Parity v2: Technical Methodological Blueprint & Execution Plan

## 1. Executive Summary & Clinical Philosophy

Traditional algorithmic fairness frameworks in healthcare often optimize static parity metrics (e.g., Equalized Odds or Demographic Parity) at a fixed classification threshold. In clinical deployment, this naive optimization frequently causes **severe calibration degradation, arbitrary risk-score distortion, or unacceptable loss of overall discriminative utility**.

In 30-day home health readmission risk prediction, a **False Negative** (failing to flag a high-risk patient who gets readmitted) carries severe clinical harm, whereas a False Positive leads to benign clinical review.

### Primary Clinical Mandate
> **Reduce worst-group missed-risk harm ($\text{FNR}_{\text{worst}}$) subject to strict clinical calibration bounds ($\text{Brier Score}$, Platt-calibrated slope $\in [0.9, 1.1]$) and discriminative retention ($\text{ROC-AUC} \ge 0.97 \times \text{ROC-AUC}_{\text{baseline}}$).**

---

## 2. Technical Architecture: Tokenized Cross-Attention Network (`ACTParityV2`)

### A. Explicit Multi-Tier Tokenization
Rather than flattening all features into single projections, **ACT-Parity v2** maps features into three distinct token sequences:

$$\mathbf{E}_{\text{Micro}} \in \mathbb{R}^{B \times N_{\text{micro}} \times D}, \quad \mathbf{E}_{\text{Meso}} \in \mathbb{R}^{B \times N_{\text{meso}} \times D}, \quad \mathbf{E}_{\text{Macro}} \in \mathbb{R}^{B \times N_{\text{macro}} \times D}$$

1. **Micro Tokens ($N_{\text{micro}}$)**: Patient clinical utilization (`Days_Cared_For`), Charlson/Elixhauser comorbidities, ICD-10 hierarchy flags (G0/G1/G2), HICD-BERT sequence embeddings, and demographic attributes.
2. **Meso Tokens ($N_{\text{meso}}$)**: Area Deprivation Index (ADI), RUCA rurality codes, census-tract educational attainment, broadband availability, housing overcrowding, and local healthcare access indicators.
3. **Macro Tokens ($N_{\text{macro}}$)**: Care process indicators, facility resource tiers, episode timing, and aggregated system capabilities.

### B. Memorization Guardrail
Direct high-cardinality agency or facility identifiers (e.g. `Agency_Medicare_Number`, `Facility_Internal_ID`) can encourage models to memorize agency-specific billing patterns rather than generalizable risk factors. These identifiers are **stripped or replaced by aggregated, defensible facility quality metrics**.

### C. Genuine Query-Key-Value Cross-Attention & Residual Anchoring
Clinical health status ($\mathbf{E}_{\text{Micro}}$) acts as the primary anchor for readmission risk. Structural neighborhood (Meso) and system (Macro) tokens modulate—but do not replace—the clinical representation.

```
                    ┌─────────────────────────┐
                    │ Micro (Clinical) Tokens │
                    └────────────┬────────────┘
                                 │ Query (Q)
                                 ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│ Meso & Macro Context    ├──►│  Cross-Attention Bottleneck │
│ Tokens (Keys & Values)  │   └──────────────┬──────────────┘
└─────────────────────────┘                  │ Attn Output
                                             ▼
                                  ┌────────────────────┐
                                  │ Context Gate g(x)  │
                                  └──────────┬─────────┘
                                             │
                                             ▼
                 h_micro ───────────► ( + ) ◄── α · (g ⊙ Attn_Out)
                                       │
                                       ▼
                            Final Risk Prediction
```

#### Mathematical Formulation
1. **Context Representation**: Concatenate structural tokens $\mathbf{C} = [\mathbf{E}_{\text{Meso}} \,\|\, \mathbf{E}_{\text{Macro}}] \in \mathbb{R}^{B \times (N_{\text{meso}} + N_{\text{macro}}) \times D}$.
2. **Cross-Attention Output**:
   $$\mathbf{A}_{\text{context}} = \text{MultiHeadAttention}\left(Q = \mathbf{E}_{\text{Micro}}, \, K = \mathbf{C}, \, V = \mathbf{C}\right)$$
3. **Contextual Gate**:
   $$\mathbf{g} = \sigma\left(\mathbf{W}_{\text{gate}} \left[ \bar{\mathbf{e}}_{\text{Micro}} \,\|\, \bar{\mathbf{c}} \right] \right) \in \mathbb{R}^{B \times 1}$$
4. **Residual Clinical Anchor Path**:
   $$\mathbf{h}_{\text{final}} = \bar{\mathbf{e}}_{\text{Micro}} + \alpha \cdot \mathbf{g} \odot \bar{\mathbf{a}}_{\text{context}}$$
   where $0 \le \alpha \le 1$ is a hyperparameter tuned on validation data.

---

## 3. Differentiable D-GAP with Augmented Lagrangian & Invariance Regularization

### A. Differentiable Soft FNR Formulation
To support end-to-end gradient-based optimization, D-GAP replaces non-differentiable step counts with soft predictions $\sigma(z_i)$:

$$\widetilde{\text{FNR}}_g = 1 - \frac{1}{n_g^+} \sum_{i: y_i=1, a_i=g} \sigma(z_i)$$

where $n_g^+ = \sum_{i: a_i=g} y_i$ is the number of positive readmission cases in group $g$.

### B. Pre-Specified Clinical Tolerance Constraint
Safety bounds are enforced against the population FNR plus a pre-registered clinical tolerance $\delta \in [0.03, 0.05]$:

$$c_g = \widetilde{\text{FNR}}_g - \left( \widetilde{\text{FNR}}_{\text{all}} + \delta \right) \le 0$$

### C. Support Thresholding & Hierarchical Shrinkage
To prevent optimization instability from small demographic strata (e.g. AIAN or NHPI), constraints $c_g$ are enforced **only if $n_g^+ \ge m$** ($m = 30$). For sparse groups, constraints shrink towards the parent population penalty:

$$\hat{c}_g = \frac{n_g^+}{n_g^+ + m_0} c_g + \left(1 - \frac{n_g^+}{n_g^+ + m_0}\right) c_{\text{parent}}$$

### D. Augmented Lagrangian Objective (`AugmentedLagrangianDGAPLoss`)
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BCE}} + \sum_{g \in G_{\text{supported}}} \left[ \lambda_g c_g + \frac{\rho}{2} \max(0, c_g)^2 \right] + \lambda_{\text{inv}} \mathcal{L}_{\text{inv}}$$

* **Epoch-Level Dual Update**: Multipliers update over full validation evaluations rather than noisy minibatches:
  $$\lambda_g^{(t+1)} \leftarrow \max\left(0, \, \lambda_g^{(t)} + \eta \cdot c_g^{(t)}\right)$$
* **Tier Invariance Regularization ($\mathcal{L}_{\text{inv}}$)**: Penalizes prediction shifts under non-essential structural perturbations:
  $$\mathcal{L}_{\text{inv}} = \frac{1}{B} \sum_{i=1}^B \left\| \sigma(f(\mathbf{x}_i^{\text{micro}}, \mathbf{x}_i^{\text{meso}}, \mathbf{x}_i^{\text{macro}})) - \sigma(f(\mathbf{x}_i^{\text{micro}}, \widetilde{\mathbf{x}}_i^{\text{meso}}, \widetilde{\mathbf{x}}_i^{\text{macro}})) \right\|^2$$

---

## 4. Robust Audit Metrics & Sensitivity Framework

### A. Stabilized Log Cross-Tier Disparity Index ($\text{CTDI}_g$)
$$\text{CTDI}_g = \log\left( \frac{\overline{A}_{\text{meso},g} + \overline{A}_{\text{macro},g} + \epsilon}{\overline{A}_{\text{micro},g} + \epsilon} \right)$$
* Prevents ratio explosion when Micro attention is small.

### B. Harm-Weighted Excess FNR Index ($\text{EFNHI}^*$)
$$\text{EFNHI}^* = \left[ \max_{g: n_g^+ \ge m} \left( \widetilde{\text{FNR}}_g - \widetilde{\text{FNR}}_{\text{all}} \right)_+ \right] \cdot \left( \sum_{g \in G} w_g \widetilde{\text{FNR}}_g \right) \cdot \exp\left( \beta \cdot \text{GEI}_{\alpha=2} \right)$$
* Penalizes excess subgroup false negative harm weighted by the Generalized Entropy Index ($\text{GEI}_{\alpha=2}$).

### C. Platt Calibration Slope & Brier Score
Fits a logistic calibration curve $\text{logit}(p) = a \cdot z + b$. A slope $a \in [0.90, 1.10]$ confirms well-calibrated risk probabilities across subgroups.

### D. Multi-Criteria Fairness-Utility Pareto Frontier
Candidate variants must satisfy pre-registered clinical acceptability criteria:
1. $\text{ROC-AUC}_{\text{model}} \ge 0.97 \times \text{ROC-AUC}_{\text{baseline}}$
2. Worst-Group FNR Gap ($\text{FNR}_{\text{worst}} - \text{FNR}_{\text{all}} \le 0.05$)
3. Platt Calibration Slope $\in [0.90, 1.10]$ across all groups with $n_g^+ \ge 30$.

### E. Structural Context Sensitivity Analysis (SCSA)
Informal causal claims are replaced with **Structural Context Sensitivity Analysis (SCSA)** using conditional reference resampling of Meso/Macro features to quantify structural attribution gap $\Delta_{\text{SDOH}}$.

---

## 5. Full Factorial Ablation Matrix (V1 to V8 Benchmark)

| Variant ID | Architecture & Feature Scope | Fairness Intervention | Target Evaluation Purpose |
| :--- | :--- | :--- | :--- |
| **V1_Micro_Only** | Micro Clinical Features Only | None | Utility Floor & Clinical Signal Baseline |
| **V2_Micro_Meso** | Micro + Meso (SDOH) | None | Incremental SDOH Predictive Value |
| **V3_Flat_Transformer** | All Tiers (No Gate, $\alpha=0.0$) | None | Flat Tabular Transformer Baseline |
| **V4_Gated_CrossAttn** | All Tiers + Gated Cross-Attn ($\alpha=0.5$) | None | Value of Residual Gated Architecture |
| **V5_Fixed_Equal_Opp** | All Tiers + Gate | Static Equal Opportunity Loss ($\delta=0.0$) | Static vs Adaptive Penalty Comparison |
| **V6_ACT_Parity_v2_Full**| All Tiers + Gate | **Augmented Lagrangian D-GAP ($\delta=0.04$)** | **Full Primary Proposed Model** |
| **V7_Tier_Invariance** | All Tiers + Gate + $\mathcal{L}_{\text{inv}}$ | Augmented Lagrangian D-GAP | Impact of Tier Invariance Regularization |
| **V8_No_Agency_IDs** | All Tiers (No Raw Agency IDs) | Augmented Lagrangian D-GAP | Agency Memorization Guardrail Audit |

---

## 6. Statistical Reporting Standard

All subgroup metrics, FNR gaps, Brier scores, and Pareto frontier coordinates are evaluated over **1,000 Stratified Patient-Level Bootstrap Resamples**, reporting 95% Confidence Intervals:

$$\text{Metric}_{\text{Reported}} = \text{Mean} \quad \left(95\%\text{ CI: } [\text{Percentile}_{2.5\%}, \, \text{Percentile}_{97.5\%}]\right)$$

---

## 7. Execution Roadmap & File Map

* **`parity/models.py`**: Implementation of `ACTParityV2` and `get_m3_feature_groups_tokenized()`.
* **`parity/loss.py`**: Implementation of `AugmentedLagrangianDGAPLoss` and `DualMultiplierManager`.
* **`parity/metrics.py`**: Implementation of stabilized CTDI, harm-weighted $\text{EFNHI}^*$, Platt calibration slope, Pareto frontier, SCSA, and 1,000-resample patient bootstrap.
* **`parity/run_experiments.py`**: Execution script for V1–V8 Factorial Ablation Matrix with parallel cohort processing.
* **`parity/run_parity.slurm`**: SLURM execution script for GPU nodes (`gpu1`).
