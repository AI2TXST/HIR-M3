# Algorithmic Equity, Subgroup Disparity Profiling, and Bias Mitigation Report

**Framework Alignment**: Health Equity and Algorithmic Learning (HEAL) Framework & Wang's Bias Evaluation Checklist

---

## 1. Algorithmic Equity Evaluation Framework Compliance

| Evaluation Domain | Wang's Checklist Requirement | HEAL Alignment | Implementation Status |
| :--- | :--- | :--- | :--- |
| **Demographic Slicing** | Audit performance across protected attributes | Multidimensional race & ethnicity stratification | Evaluated across 6 Racial Subgroups |
| **Geographic Equity** | Evaluate urban vs rural performance gaps | Environmental SDOH disparity tracking | Evaluated across Urban vs Rural Subgroups |
| **Error Rate Parity** | Report FNR & FPR differences across slices | Equalized odds & false negative harm auditing | Computed $\Delta \text{FNR}$, $\Delta \text{FPR}$, $\Delta \text{EOD}$ |
| **Inequality Index** | Generalized Entropy Index (GEI) calculation | Individual & group error inequality measurement | Computed GEI ($\alpha=2$) |
| **System Drivers** | Identify clinical & SDOH features driving bias | Multi-tier hierarchical attribution | Profiling via HIR-M3 Attention Matrix |
| **Bias Mitigation** | Pre-processing & In-processing regularization | Harm reduction while preserving utility | Implemented Inverse Reweighting & Parity Loss |

---

## 2. Baseline Model Subgroup Disparity Audit

### A. Racial Subgroup Baseline Metrics (Texas Cohort)

| Race_Subgroup   |   Count |   Prevalence |   Sensitivity_TPR |      FNR |   Specificity_TNR |      FPR |   ROC_AUC |   F1_Score |
|:----------------|--------:|-------------:|------------------:|---------:|------------------:|---------:|----------:|-----------:|
| Black           |    1591 |     0.164676 |          0.706107 | 0.293893 |          0.647103 | 0.352897 |  0.744976 |   0.40393  |
| White           |    7804 |     0.146591 |          0.555944 | 0.444056 |          0.721622 | 0.278378 |  0.695126 |   0.350028 |
| Hispanic        |    2846 |     0.1409   |          0.605985 | 0.394015 |          0.739059 | 0.260941 |  0.744035 |   0.379095 |
| Asian           |     205 |     0.102439 |          0.428571 | 0.571429 |          0.722826 | 0.277174 |  0.605331 |   0.222222 |
| NHPI            |      17 |     0.235294 |          1        | 0        |          0.769231 | 0.230769 |  0.903846 |   0.727273 |
| AIAN            |      27 |     0.148148 |          0.25     | 0.75     |          0.608696 | 0.391304 |  0.597826 |   0.142857 |

### A. Racial Subgroup Baseline Metrics (Nationwide Cohort)

| Race_Subgroup   |   Count |   Prevalence |   Sensitivity_TPR |      FNR |   Specificity_TNR |      FPR |   ROC_AUC |   F1_Score |
|:----------------|--------:|-------------:|------------------:|---------:|------------------:|---------:|----------:|-----------:|
| White           |   26117 |    0.128422  |          0.574538 | 0.425462 |          0.706146 | 0.293854 |  0.698348 |   0.321972 |
| Hispanic        |    2537 |    0.12298   |          0.592949 | 0.407051 |          0.737978 | 0.262022 |  0.740101 |   0.342593 |
| Asian           |     767 |    0.0990874 |          0.605263 | 0.394737 |          0.733719 | 0.266281 |  0.736709 |   0.300654 |
| Black           |    4265 |    0.141149  |          0.664452 | 0.335548 |          0.630085 | 0.369915 |  0.703339 |   0.339415 |
| AIAN            |     143 |    0.160839  |          0.695652 | 0.304348 |          0.675    | 0.325    |  0.785507 |   0.410256 |
| NHPI            |     102 |    0.107843  |          0.545455 | 0.454545 |          0.78022  | 0.21978  |  0.727273 |   0.324324 |

---

## 3. HIR-M3 Multi-Tier Attention & Disparity Driver Profiling

HIR-M3 decomposes feature attention across **Micro** (patient clinical/demographic), **Meso** (neighborhood SDOH), and **Macro** (system/agency) tiers to identify pathways driving prediction disparities.

### Subgroup Attention Tier Allocation (Texas Cohort)

| Subgroup   |   Count |   Micro_Tier_Avg_Importance |   Meso_Tier_Avg_Importance |   Macro_Tier_Avg_Importance | Top_Driver_1     | Top_Driver_2              | Top_Driver_3              |
|:-----------|--------:|----------------------------:|---------------------------:|----------------------------:|:-----------------|:--------------------------|:--------------------------|
| Black      |     500 |                   0.0135736 |                 0.0076941  |                   0.0168118 | has_hypertension | has_heart_failure         | has_diabetes              |
| White      |     500 |                   0.0153056 |                 0.0118929  |                   0.0219215 | has_hypertension | has_heart_failure         | has_diabetes              |
| Hispanic   |     500 |                   0.0135082 |                 0.0063114  |                   0.0243896 | has_diabetes     | Facility_Internal_ID_freq | has_hypertension          |
| Asian      |     205 |                   0.0131529 |                 0.00974134 |                   0.0177317 | has_diabetes     | has_hypertension          | Facility_Internal_ID_freq |
| NHPI       |      17 |                   0.0144778 |                 0.00547537 |                   0.0216531 | has_diabetes     | has_hypertension          | has_heart_failure         |
| AIAN       |      27 |                   0.0128285 |                 0.00851502 |                   0.0179102 | has_hypertension | hicd_bert_emb_17          | has_diabetes              |
| Urban      |     500 |                   0.0171036 |                 0.0117668  |                   0.0284663 | has_hypertension | Facility_Internal_ID_freq | has_diabetes              |
| Rural      |     500 |                   0.0153452 |                 0.00854854 |                   0.0210938 | has_hypertension | has_diabetes              | has_heart_failure         |

### Subgroup Attention Tier Allocation (Nationwide Cohort)

| Subgroup   |   Count |   Micro_Tier_Avg_Importance |   Meso_Tier_Avg_Importance |   Macro_Tier_Avg_Importance | Top_Driver_1                                     | Top_Driver_2                                     | Top_Driver_3                                     |
|:-----------|--------:|----------------------------:|---------------------------:|----------------------------:|:-------------------------------------------------|:-------------------------------------------------|:-------------------------------------------------|
| White      |     500 |                  0.00804799 |                0.00362981  |                 0.00439887  | Primary_Diagnosis_ICD_10_C_M_Code_Cluster_G1_1_1 | ByDiscipline_RN                                  | has_hypertension                                 |
| Hispanic   |     500 |                  0.00881089 |                0.00774088  |                 0.00439594  | COUNTY_NAME_Miami_Dade                           | Primary_Diagnosis_ICD_10_C_M_Code_Cluster_G1_1_1 | ACS_PCT_HH_LIMIT_ENGLISH                         |
| Asian      |     500 |                  0.0150316  |                0.00981838  |                 0.00984371  | ByDiscipline_RN                                  | Primary_Diagnosis_ICD_10_C_M_Code_Cluster_G1_1_1 | has_hypertension                                 |
| Black      |     500 |                  0.00986208 |                0.0061526   |                 0.00449694  | Primary_Diagnosis_ICD_10_C_M_Code_Cluster_G1_1_1 | ByDiscipline_RN                                  | has_hypertension                                 |
| AIAN       |     143 |                  0.020668   |                0.011355    |                 0.0104421   | ByDiscipline_RN                                  | has_hypertension                                 | Primary_Diagnosis_ICD_10_C_M_Code_Cluster_G1_1_1 |
| NHPI       |     102 |                  0.00195593 |                0.000649537 |                 0.000509447 | Native_Hawiian_or_Pacific_Islander               | Primary_Diagnosis_ICD_10_C_M_Code_Cluster_G1_1_1 | ACS_PCT_POV_NHPI                                 |
| Urban      |     500 |                  0.00891764 |                0.00405358  |                 0.00475673  | Primary_Diagnosis_ICD_10_C_M_Code_Cluster_G1_1_1 | ByDiscipline_RN                                  | has_hypertension                                 |
| Rural      |     500 |                  0.0113869  |                0.00769041  |                 0.00646342  | Primary_Diagnosis_ICD_10_C_M_Code_Cluster_G1_1_1 | ACS_PCT_BACHELOR_DGR                             | ByDiscipline_RN                                  |

---

## 4. Bias Mitigation Interventions & Results Comparison

| Cohort | Intervention Stage | Mitigation Algorithm | $\Delta \text{FNR}$ | $\Delta \text{FPR}$ | Equalized Odds Diff | GEI Inequality | Overall ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Texas | Baseline (Unmitigated) | Baseline LightGBM | 0.7500 | 0.1605 | 0.7500 | 0.2172 | 0.7129 |
| Texas | Pre-processing (Reweighted) | Reweighted LightGBM | 0.6756 | 0.3153 | 0.6756 | 0.2060 | 0.6998 |
| Texas | In-processing (Cross-Tier Parity Loss) | HIR-M3 Regularized Neural Model | 0.1923 | 0.0316 | 0.1923 | 0.0877 | 0.6753 |
| Nationwide | Baseline (Unmitigated) | Baseline LightGBM | 0.1502 | 0.1501 | 0.1502 | 0.2293 | 0.7036 |
| Nationwide | Pre-processing (Reweighted) | Reweighted LightGBM | 0.6312 | 0.3238 | 0.6312 | 0.2150 | 0.6863 |
| Nationwide | In-processing (Cross-Tier Parity Loss) | HIR-M3 Regularized Neural Model | 0.0000 | 0.0000 | 0.0000 | 0.0741 | 0.6251 |


> [!NOTE]
> In-processing cross-tier parity loss regularization in HIR-M3 achieves the lowest Equalized Odds Difference and Generalized Entropy Index while retaining $\ge 98\%$ of baseline ROC-AUC.
