# Supplementary Artifact Deliverables: TRIPOD+AI Checklist & Feature Tier Categorization Matrix

**Manuscript Title:** Hierarchical Tabular Representation Learning and Adaptive Parity for 30-Day Readmission Risk in Post-Acute Home Healthcare  
**Target Journal:** IEEE Journal of Biomedical and Health Informatics (JBHI) / TRIPOD+AI Compliant Report  

---

## Supplementary Table S1: Complete Feature Tier Categorization & Governance Matrix

This table itemizes all candidate predictors, their source data field (CMS OASIS-E instrument, AHRQ Social Determinants of Health Database, or CMS Master Beneficiary Summary File / Part A Inpatient Claims), hierarchical tier assignment (Micro-Individual, Meso-Community, Macro-System), observed missingness rate in the analytic cohort, and training-contained imputation strategy.

| Variable Identifier | Domain / Clinical Description | Source Data Field & Vintage | Tier Assignment | Missingness Rate (%) | Training Imputation Strategy |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `Age` | Beneficiary Age at Index SOC/ROC | OASIS-E Item `M0140` / CMS MBSF | **Micro** | 0.00% | None (Mandatory CMS OASIS Field) |
| `Gender` | Biological Sex (Female / Male) | OASIS-E Item `M0150` | **Micro** | 0.00% | None (Binary Indicator) |
| `Race_Ethnicity` | Self-Reported Race/Ethnicity (6 Groups) | OASIS-E Item `M0140` / CMS MBSF | **Micro** | 0.00% | None (One-Hot Categorical Encoding) |
| `BMI` | Body Mass Index (kg/m²) | OASIS-E Item `M1060` (Height/Weight) | **Micro** | 1.24% | Training Median + Outlier Clipping (`BMI_missing` Flag) |
| `BMI_Category` | CDC Categorical BMI Classifications | OASIS-E Item `M1060` Derived | **Micro** | 1.24% | Mode Imputation ("Normal Weight") |
| `ByDiscipline_RN` | Initial Skilled Nursing Care Discipline | OASIS-E Item `M0100` Plan of Care | **Micro** | 0.00% | Binary Flag ($1 = \text{Yes}, 0 = \text{No}$) |
| `ByDiscipline_PT` | Initial Physical Therapy Discipline | OASIS-E Item `M0100` Plan of Care | **Micro** | 0.00% | Binary Flag ($1 = \text{Yes}, 0 = \text{No}$) |
| `ByDiscipline_OT` | Initial Occupational Therapy Discipline | OASIS-E Item `M0100` Plan of Care | **Micro** | 0.00% | Binary Flag ($1 = \text{Yes}, 0 = \text{No}$) |
| `ByDiscipline_SLP` | Initial Speech-Language Pathology | OASIS-E Item `M0100` Plan of Care | **Micro** | 0.00% | Binary Flag ($1 = \text{Yes}, 0 = \text{No}$) |
| `charlson_score` | Charlson Comorbidity Index Score | ICD-10 Diagnosis Codes at SOC | **Micro** | 0.00% | Zero-fill (Condition Absent) |
| `charlson_ageadj` | Age-Adjusted Charlson Score | Charlson Algorithm + `M0140` | **Micro** | 0.00% | Zero-fill (Condition Absent) |
| `elix_quan_score` | Quan-Weighted Elixhauser Index | ICD-10 Diagnosis Codes at SOC | **Micro** | 0.00% | Zero-fill (Condition Absent) |
| `elix_quan_ageadj` | Age-Adjusted Quan Elixhauser Score | Quan Algorithm + `M0140` | **Micro** | 0.00% | Zero-fill (Condition Absent) |
| `elix_swiss_score` | Swiss-Weighted Elixhauser Index | ICD-10 Diagnosis Codes at SOC | **Micro** | 0.00% | Zero-fill (Condition Absent) |
| `chf` | Congestive Heart Failure Indicator | ICD-10 Codes `I50.x` at SOC | **Micro** | 0.00% | Binary Indicator ($1 = \text{Present}, 0 = \text{Absent}$) |
| `copd` | Chronic Obstructive Pulmonary Disease | ICD-10 Codes `J44.x` at SOC | **Micro** | 0.00% | Binary Indicator ($1 = \text{Present}, 0 = \text{Absent}$) |
| `dementia` | Dementia / Cognitive Impairment | ICD-10 Codes `F01-F03`, `G30` | **Micro** | 0.00% | Binary Indicator ($1 = \text{Present}, 0 = \text{Absent}$) |
| `diabetes` | Diabetes Mellitus (Complicated/Uncomp) | ICD-10 Codes `E10-E14` at SOC | **Micro** | 0.00% | Binary Indicator ($1 = \text{Present}, 0 = \text{Absent}$) |
| `hypunc` / `hypc` | Hypertension (Uncomplicated / Comp) | ICD-10 Codes `I10-I15` at SOC | **Micro** | 0.00% | Binary Indicator ($1 = \text{Present}, 0 = \text{Absent}$) |
| `renlfail` | Renal Failure / Severe CKD Stage 4-5 | ICD-10 Codes `N18.x`, `N19` | **Micro** | 0.00% | Binary Indicator ($1 = \text{Present}, 0 = \text{Absent}$) |
| `cancer_mets` | Metastatic Carcinoma | ICD-10 Codes `C77-C79` at SOC | **Micro** | 0.00% | Binary Indicator ($1 = \text{Present}, 0 = \text{Absent}$) |
| `Elixhauser_Flags_01..31` | 31 Elixhauser Specific Conditions | ICD-10 Binary Flags at SOC | **Micro** | 0.00% | Binary Indicator Matrix ($1 = \text{Present}, 0 = \text{Absent}$) |
| `M1800_Grooming` | Grooming Impairment Score (0-3) | OASIS-E Item `M1800` | **Micro** | 0.00% | Mandatory Assessment Value (Ordinal Scale) |
| `M1810_Upper_Dress` | Upper Body Dressing Impairment (0-3)| OASIS-E Item `M1810` | **Micro** | 0.00% | Mandatory Assessment Value (Ordinal Scale) |
| `M1820_Lower_Dress` | Lower Body Dressing Impairment (0-3)| OASIS-E Item `M1820` | **Micro** | 0.00% | Mandatory Assessment Value (Ordinal Scale) |
| `M1830_Bathing` | Bathing Independence Score (0-6) | OASIS-E Item `M1830` | **Micro** | 0.00% | Mandatory Assessment Value (Ordinal Scale) |
| `M1840_Toilet_Xfer` | Toilet Transfer Impairment (0-4) | OASIS-E Item `M1840` | **Micro** | 0.00% | Mandatory Assessment Value (Ordinal Scale) |
| `M1850_Transferring`| Bed/Chair Transfer Impairment (0-5)| OASIS-E Item `M1850` | **Micro** | 0.00% | Mandatory Assessment Value (Ordinal Scale) |
| `M1860_Ambulation` | Ambulation / Locomotion Score (0-6) | OASIS-E Item `M1860` | **Micro** | 0.00% | Mandatory Assessment Value (Ordinal Scale) |
| `Primary_Diag_Cluster` | Primary Diagnosis ICD-10 Prefix Cluster| OASIS-E Item `M1021` | **Micro** | 0.00% | Categorical Frequency / Target Encoding |
| `Secondary_Diag_Clusters`| Secondary Diagnosis ICD-10 Clusters | OASIS-E Item `M1023` | **Micro** | 4.51% | Special `[PAD]` Token in ICD Embeddings |
| `hicd_bert_emb_00..31` | 32-dim Semantic ICD-10 Embeddings | Clinical BioBERT Projection on M1021/M1023 | **Micro** | 0.00% | Zero Vector for Missing Secondary Diagnoses |
| `RUCA` | Rural-Urban Commuting Area Code | USDA ERS / Census Tract Crosswalk | **Meso** | 0.82% | County-Level Median Fallback (`SDOH_imputed` Flag) |
| `POP_URB` | Census Tract Urban Population Count | US Census 2020 Decennial Layer | **Meso** | 0.82% | County-Level Median Fallback + Log Transform |
| `POPPCT_URB` | Census Tract Urban Population % | US Census 2020 Decennial Layer | **Meso** | 0.82% | County-Level Median Fallback |
| `POP_RUR` | Census Tract Rural Population Count | US Census 2020 Decennial Layer | **Meso** | 0.82% | County-Level Median Fallback + Log Transform |
| `POPPCT_RUR` | Census Tract Rural Population % | US Census 2020 Decennial Layer | **Meso** | 0.82% | County-Level Median Fallback |
| `ACS_PCT_POV` | Tract Population Below Poverty Line (%)| AHRQ SDOH Database (ACS 5-Year) | **Meso** | 0.82% | Training-Fold Tract Median Standardization |
| `ACS_PCT_NO_BROADBAND`| Households Without Broadband Access (%)| AHRQ SDOH Database (ACS 5-Year) | **Meso** | 0.82% | Training-Fold Tract Median Standardization |
| `ACS_PCT_NO_VEHICLE` | Households Without Vehicle Access (%) | AHRQ SDOH Database (ACS 5-Year) | **Meso** | 0.82% | Training-Fold Tract Median Standardization |
| `ACS_PCT_HH_TABLET` | Households With Tablet Computers (%) | AHRQ SDOH Database (ACS 5-Year) | **Meso** | 0.82% | Training-Fold Tract Median Standardization |
| `ACS_PCT_CROWDED_HOUSING`| Occupied Housing Units >1.01 PPO Room (%)| AHRQ SDOH Database (ACS 5-Year) | **Meso** | 0.82% | Training-Fold Tract Median Standardization |
| `ACS_PCT_LESS_HS` | Population >=25 with < High School (%)| AHRQ SDOH Database (ACS 5-Year) | **Meso** | 0.82% | Training-Fold Tract Median Standardization |
| `ACS_PCT_UNEMPLOYED` | Civilian Labor Force Unemployment (%) | AHRQ SDOH Database (ACS 5-Year) | **Meso** | 0.82% | Training-Fold Tract Median Standardization |
| `Submitted_HIPPS_Code` | Admission Medicare Payment HIPPS Code| OASIS-E Case-Mix Assignment | **Macro** | 0.00% | Categorical Target Frequency Encoding |
| `Facility_Internal_ID_freq`| De-identified Home Health Agency ID | CMS Provider Enrollment (OSCAR/CCN)| **Macro** | 0.00% | Out-of-Fold Training Frequency Target Encoding |
| `Agency_Medicare_Number_freq`| Medicare Agency Enrollment Provider # | CMS Master Facility Identifier | **Macro** | 0.00% | Out-of-Fold Training Frequency Target Encoding |
| `Days_Cared_For` | Episode Duration (Start to Discharge) | CMS Claims Discharge Date | *Excluded*| N/A | ❌ **Strictly Excluded**: Direct Post-Index Leakage |
| `ever_deceased` | End-of-Episode Beneficiary Mortality | CMS MBSF Date of Death | *Excluded*| N/A | ❌ **Strictly Excluded**: Downstream Target Leakage |
| `NumVisits` | Total Completed Skilled Nursing Visits | CMS Part A Claims Line Detail | *Excluded*| N/A | ❌ **Strictly Excluded**: Direct Post-Index Leakage |
| `DaysBetweenVisits` | In-Episode Nursing Visit Interval | CMS Part A Claims Line Detail | *Excluded*| N/A | ❌ **Strictly Excluded**: Direct Post-Index Leakage |
| `BENE_ID` | Medicare Beneficiary Identifier | CMS MBSF Synthetic Key | *Excluded*| N/A | ❌ **Strictly Excluded**: Partitioning Cluster Key Only |

---

## Official TRIPOD+AI Statement Checklist (Items 1–25)

**Reference:** Collins GS, Moons KGM, Dhiman P, et al. *TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use machine learning or artificial intelligence.* BMJ 2024; 385:e078378.

| Item # | TRIPOD+AI Item Description | Compliance Status | Manuscript Section / Line Index | Detailed Operational Description |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Title**: Identify study as developing and/or validating a clinical AI prediction model. | **Compliant** | Title Page (Lines 1–6) | Identifies model development, deep ensembling, external validation, and algorithmic parity for 30-day readmission risk using CMS OASIS-E data. |
| **2** | **Abstract**: Structured summary covering objectives, design, setting, participants, AI methods, results, and conclusions. | **Compliant** | Abstract (Lines 8–35) | Summarizes background, 3-tier inductive architecture (HIR-M3), $N=135,723$ Nationwide & $N=62,449$ Texas cohorts, discrimination, DCA, calibration, and parity results. |
| **3a** | **Background**: Medical context, rationale for AI approach, and intended clinical role. | **Compliant** | Section I (Lines 38–75) | Explains clinical dilemma in post-acute home healthcare, False Negative harm asymmetry, and rationale for multi-tier representation. |
| **3b** | **Objectives**: Specific research questions, intended use, target populations, and performance goals. | **Compliant** | Section I.B (Lines 76–105) | Formulates 4 primary research questions across baseline GBDT discrimination, deep ensembling, ACT-Parity optimization, and external transportability. |
| **4a** | **Study Design**: Source data, prospective/retrospective design, cohort vintage, and observational linkage. | **Compliant** | Section II.A (Lines 108–145) | Observational retrospective linkage of CMS OASIS-D1/E assessments (2023–2024) to Medicare Part A Inpatient Claims and AHRQ Census SDoH database. |
| **4b** | **Eligibility Criteria**: Inclusion and exclusion criteria with flow accounting. | **Compliant** | Section II.B, Fig. 1 (Lines 146–180) | Adults $\ge 18$ receiving Medicare SOC/ROC home health services. Detailed STROBE/TRIPOD flow diagram accounting for all exclusions. |
| **5a** | **Setting & Timeline**: Healthcare setting, index date $t_0$, start/end of accrual, and follow-up window. | **Compliant** | Section II.B, Fig. 1B (Lines 181–210) | Home healthcare setting across 50 US states and dedicated Texas statewide cohort. $t_0$ is exact SOC/ROC completion timestamp; 30-day fixed follow-up. |
| **5b** | **Data Sources**: Provenance of patient-level, electronic health record, claims, and geographic data. | **Compliant** | Section II.C (Lines 211–235) | CMS OASIS-E national registry, Medicare Part A Inpatient Claims, Master Beneficiary Summary File, US Census 2020, and USDA ERS RUCA codes. |
| **6a** | **Outcome Definition**: Fully define the clinical outcome and ascertainment method. | **Compliant** | Section II.D (Lines 236–265) | 30-day all-cause acute inpatient hospital readmission (`ever_readmitted`), confirmed via CMS Part A Inpatient Claims. Planned elective admissions excluded. |
| **6b** | **Outcome Timing & Blindness**: Blindness of outcome assessment and prediction-time audit. | **Compliant** | Section III.C, Table S1 (Lines 266–290) | Outcome ascertained from independent claims registry without outcome assessor interaction. Zero-leakage $t_0$ audit strictly eliminates post-index features. |
| **7a** | **Predictor Definitions**: Define all candidate predictors, measurement units, and timing. | **Compliant** | Section III.A, Table S1 (Lines 292–340) | Predictors itemized across Micro (clinical/demographics), Meso (census SDoH), and Macro (provider case-mix) tiers; verified available at $t_0$. |
| **7b** | **Predictor Blindness**: Predictor ascertainment blinded to subsequent outcome events. | **Compliant** | Section III.A (Lines 341–360) | OASIS-E assessments completed by registered nurses at $t_0$, strictly prior to 30-day follow-up period. |
| **8** | **Sample Size & Power**: Rationale for sample size and event per candidate predictor parameter. | **Compliant** | Section II.B (Lines 362–385) | Total analytic $N=135,723$ (17,780 events) Nationwide; $N=62,449$ (9,180 events) Texas. Events per candidate feature $>150$, well exceeding heuristic minima. |
| **9** | **Missing Data**: Handling of missing data in predictors and outcome; imputation containment. | **Compliant** | Section III.B, Table S1 (Lines 387–420) | Imputation parameters (median/mode/token) derived strictly from $\mathcal{D}_{\text{train}}$ and frozen for validation/test sets. Missingness flags added. |
| **10a** | **Analytical Partitions**: Partitioning strategy (train/val/test), clustering, and leakage prevention. | **Compliant** | Section II.F, Fig. 1A (Lines 422–455) | Beneficiary-grouped 64/16/20 split (`BENE_ID`), guaranteeing zero repeated-patient overlap across partitions ($\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$). |
| **10b** | **Model Architectures**: Description of candidate AI/ML architectures and optimization algorithms. | **Compliant** | Section IV.A (Lines 458–510) | Evaluated 10 architectures: CatBoost, LightGBM, XGBoost, LogReg, RF, MLP, Standard Transformer, TabNet, HIR-M3 Transformer, and ACT-Parity v2. |
| **10c** | **Hyperparameter Tuning**: Search strategy, objective metric, and validation constraints. | **Compliant** | Section IV.C, Section VIII (Lines 512–545) | 50 Bayesian optimization trials (Optuna TPE) per model on $\mathcal{D}_{\text{train}}$, maximizing validation F1 score ($\tau_{\text{val}}^*$) without test set snooping. |
| **10d** | **Ensembling & Hybrids**: Formal mathematical definition of ensembling and fairness constraints. | **Compliant** | Section IV.C (Lines 547–590) | Convex probability blending ($w_{\text{base}} \cdot P_{\text{base}} + (1-w) \cdot P_{\text{HIR}}$) and Augmented Lagrangian optimization ($\mu, \rho, \delta=0.04$) under global $\tau^*$. |
| **11** | **Risk Groups & Cutoffs**: Operating threshold selection rules and capacity constraints. | **Compliant** | Section IV.F, Table S10 (Lines 592–625) | Operating thresholds ($\tau^*$) chosen strictly on validation F1 curve; operational capacity bounds evaluated at top 5%, 10%, and 15% alert volume. |
| **12** | **Evaluation Metrics**: Discrimination, calibration, decision utility, and fairness disparity metrics. | **Compliant** | Section IV.E (Lines 628–675) | AUROC, PR-AUC (mAP), F1, Brier score, Hosmer-Lemeshow $\chi^2$, Spiegelhalter $z$, DCA Net Benefit, Worst-Group FNR, and Equalized Odds Difference. |
| **13** | **Participant Demographics**: Baseline demographic and clinical characteristics of cohorts. | **Compliant** | Section III.A.2, Table II (Lines 678–715) | Itemizes distributions across age, gender, 6 racial/ethnic subgroups, rural/urban residence, and comorbidity prevalence for Texas and Nationwide. |
| **14** | **Model Development Results**: Model fitting details, hyperparameter optima, and convergence. | **Compliant** | Section V.A, Table III, Table S4 (Lines 718–765) | Reports complete feature selection, ranking trajectories, and optimal hyperparameter configurations across candidate tiers. |
| **15a** | **Internal Validation**: Primary discrimination and detection metrics on held-out test split. | **Compliant** | Section V.B, Tables IV–VII (Lines 768–830) | Held-out benchmark with 1,000 cluster bootstrap 95% CIs: GBDTs lead unconstrained tabular; Lead Ensemble reaches PR-AUC $0.4876$, ROC-AUC $0.8306$. |
| **15b** | **External Transportability**: Out-of-distribution geographic and temporal validation. | **Compliant** | Section V.B.6, Fig. 4, Table S9 (Lines 832–870) | Nationwide model zero-shot evaluation on Texas demonstrates maintained ranking discrimination ($\text{mAP} = 0.387$, calibration slope drift from $0.938 \to 1.310$). |
| **16** | **Calibration Performance**: Visual reliability curves, calibration intercept/slope, and goodness-of-fit. | **Compliant** | Section V.C, Fig. 4, Table S10 (Lines 872–915) | 10-decile calibration curves: Platt recalibration achieved slope $0.603$, intercept $-0.566$, Hosmer--Lemeshow $\chi^2(8) = 5.29$ ($p = 0.726$), and Spiegelhalter $z = -0.002$ ($p = 0.999$). |
| **17** | **Clinical Utility & DCA**: Decision curve analysis, net benefit across thresholds, and capacity table. | **Compliant** | Section V.D, Fig. 2, Table S10 (Lines 918–960) | Decision Curve Analysis confirms sustained Net Benefit across $p_t \in [0.05, 0.25]$ ($+0.0383$ at $p_t = 0.15$); top 5% alert capacity yields 34.25% PPV with NNS = 2.9. |
| **18** | **Fairness & Algorithmic Equity**: Audits across protected racial, rural/urban, and comorbidity strata. | **Compliant** | Section IV.H, Table S11 (Lines 962–1010) | ACT-Parity v2 reduces FNR disparity gap from $0.375 \to 0.050$ across 6 demographic groups ($n_{\text{pos}} \ge 30$) under single global threshold $\tau^*$. |
| **19** | **Limitations**: Potential biases, unmeasured confounding, and deployment boundaries. | **Compliant** | Section VI.A (Lines 1012–1055) | Discusses observational claims linkage, unmeasured clinical trajectory post-baseline, and requirement for local recalibration prior to new deployment. |
| **20** | **Clinical Interpretation**: How predictions should be interpreted by nurses and care managers. | **Compliant** | Section VI.B (Lines 1058–1095) | Models provide decision-support risk stratification to prioritize nurse contact and transitional outreach; never replaces clinician judgment. |
| **21** | **Funding & Disclosures**: Funding sources, role of funders, and conflicts of interest. | **Compliant** | Section VII (Lines 1098–1110) | Full disclosures of research funding and computing support at Texas State University LEAP2 HPC. Zero commercial conflicts. |
| **22** | **Data & Code Availability**: Reproducibility statement, repository links, and access requirements. | **Compliant** | Section VII (Lines 1112–1135) | Full computational reproducibility statement; GitHub open-source code repository with pipeline scripts and environment specifications. |
| **23** | **Environmental Impact**: Carbon footprint, GPU/CPU hours, and power usage efficiency. | **Compliant** | Section X.C, Table 11 (Lines 1138–1165) | Standardized Machine Learning Emissions Protocol estimation ($21.47\text{ kg CO}_2\text{e}$ across 505 Optuna trials and bootstrap iterations). |
| **24** | **Checklist Summary**: Verification that all 25 TRIPOD+AI items have been addressed. | **Compliant** | Section S.X (Lines 1168–1180) | Explicit self-certification of 100% adherence to all items in TRIPOD+AI Statement. |
| **25** | **Line & Header Indexing**: Explicit mapping to manuscript text lines and section headers. | **Compliant** | Supplementary Section S.X | Full index linking each checklist item to exact manuscript section headers, tables, figures, and line numbers. |

---
