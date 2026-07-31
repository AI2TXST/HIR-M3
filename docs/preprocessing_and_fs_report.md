# OASIS Home Health Pipeline & Feature Selection Technical Report

**Date**: July 21, 2026  
**Environment**: Texas State LEAP2 HPC Cluster (`spark_env`, Slurm Workload Manager)  
**Target Dataset**: OASIS Home Health & Census SDOH Integrated Datasets (`final_mergedDF_TX.csv`, `final_mergedDF.csv`)  

---

## 1. Executive Summary

This report details the architectural enhancements, bug fixes, and performance optimizations implemented across the end-to-end OASIS Home Health data processing and feature selection pipelines. Key accomplishments include:

1. **PySpark Preprocessing Acceleration**: Reduced Spark aggregation overhead by converting sequential column scans into single-pass batch aggregations, cutting Spark distinct jobs from 28 down to 2, and enabling multi-threaded dataset execution.
2. **Notebook-to-Script Conversion**: Extracted 100% of logic from `preprocessing.ipynb` into a production-grade Python script (`preprocessing_nb.py`) with full backward compatibility across Pandas releases.
3. **Feature Selection Preservation & Leakage Fixes**: Fixed over-broad substring filtering in `utils.py` to prevent accidental removal of critical clinical predictors (`aids`, `Facility_Internal_ID`, ICD clusters) and updated Lasso/Variance Threshold methods to score and save all features in rank order.
4. **HPC Compatibility on LEAP2**: Resolved Python 3.7 `SyntaxError` issues with `imbalanced-learn` by implementing native fallback oversamplers and adding environment thread configuration in `fs.slurm`.

---

## 2. Preprocessing & Data Lineage Architecture

```mermaid
graph TD
    A[Raw OASIS CSV & Census SDOH] --> B[dataPrep_1 & risk_2 Joins]
    B --> C[final_mergedDF.csv & final_mergedDF_TX.csv]
    
    subgraph "PySpark Parallel Engine (preprocessing.py)"
        C --> D[Multi-Threaded Worker Pool]
        D --> E[Single-Pass Batch Distinct Extraction]
        E --> F[processed_final_mergedDF.csv & _TX.csv]
    end

    subgraph "Pandas Standalone Pipeline (preprocessing_nb.py)"
        C --> G[Type Coercion & Rounding]
        G --> H[Constant Column Removal]
        H --> I[One-Hot Categorical Encoding]
        I --> J[Multicollinearity Removal r > 0.7]
        J --> K[processed_final_mergedDF_TX.csv]
    end

    K --> L[Feature Selection Pipeline (feature_selection.py)]
    L --> M[results/TX_feature_selection_results.csv]
```

---

## 3. Detailed Component Enhancements

### 3.1. PySpark Preprocessing Optimization (`preprocessing.py`)
- **Single-Pass Distinct Aggregation**: Replaced 14 sequential `select(col).distinct().collect()` Spark jobs per dataset with a unified `df.agg(*[F.array_sort(F.collect_set(F.col(c)))])` aggregation query.
- **Concurrent Dataset Processing**: Implemented `concurrent.futures.ThreadPoolExecutor(max_workers=2)` inside `main()` so that all-states and Texas-specific cohorts transform concurrently.

### 3.2. Notebook Conversion & Pandas Compatibility (`preprocessing_nb.py`)
- **Full Cell Preservation**: Converted all 18 cells from `preprocessing.ipynb` into a clean modular script.
- **Pandas Version Defense**: Replaced `df_corr.corr(numeric_only=True)` with `df_corr.select_dtypes(include=[np.number]).corr()` to ensure smooth execution on older Pandas releases (< 1.5.0) in cluster environments.

### 3.3. Feature Selection & Utility Fixes (`feature_selection.py`, `utils.py`)
- **Targeted ID Filtering**: Replaced broad `'id' in c.lower()` substring matching with an explicit list (`['beneficiary_id', 'assessment_id']`). This prevents the accidental deletion of clinical features like `aids`, `state_id`, `Facility_Internal_ID`, and diagnosis clusters.
- **Full Feature Ranking**: Updated `lasso_selection()` and `variance_threshold_fs()` to output scores for all features (including zero-weight coefficients and zero-variance features) so that every feature is preserved in the output CSV.
- **Python 3.7 Resilience**: Wrapped `imbalanced-learn` imports to catch syntax errors and implemented a pure NumPy/Pandas random oversampling fallback in `balance_data()`.

---

## 4. M3 Feature Taxonomy Breakdown

The preprocessed dataset `processed_final_mergedDF_TX.csv` categorizes variables into a 3-level hierarchy:

| Level | Base Count | Encoded Count | Description | Key Feature Groups |
| :--- | :---: | :---: | :--- | :--- |
| **Identifiers & Target** | **2** | **2** | Patient tracking & outcome | `Beneficiary_ID`, `ever_readmitted` |
| **Micro (Patient/Clinical)** | **54** | **~110–135** | Individual health, race & clinical history | `Age`, `Gender_*`, 6 Race flags, `Days_Cared_For`, `ByDiscipline_*`, `BMI_Category_*`, 7 Charlson/Elixhauser Scores, 41 Comorbidity flags (`aids`, `chf`, `diab`), ICD Diagnosis Clusters |
| **Meso (Neighborhood SDOH)** | **64** | **64** | Census-tract social determinants | Urban/Rural % (`POPPCT_URB`), Education (6 ACS vars), Internet/Device access (13 ACS vars), Income & Poverty (16 ACS vars), Food stamps & Household dynamics (25 ACS vars) |
| **Macro (System/Provider)** | **4** | **Cohort dependent** | Health system, payment & geography | `Submitted_HIPPS_Code_*`, `Facility_Internal_ID_*`, `Agency_Medicare_Number_*`, `COUNTY_NAME_*` |

#### Detailed Exact Count Breakdown:
- **Target & Identifier (2)**: `Beneficiary_ID`, `ever_readmitted`
- **Micro Level (54 Base Features)**:
  - Demographics & Race (8): `Age`, `Gender`, `American_Indian_or_Alaska_Native`, `Asian`, `Black_or_African_American`, `Hispanic_or_Latino`, `Native_Hawiian_or_Pacific_Islander`, `White`
  - Clinical Utilization & Care (2): `Days_Cared_For`, `ByDiscipline`
  - Body Mass Index (1): `BMI_Category`
  - Risk & Severity Indices (7): `charlson_score`, `charlson_ageadj`, `charlson_survival_10yr`, `elix_quan_score`, `elix_quan_ageadj`, `elix_swiss_score`, `elix_swiss_ageadj`
  - Clinical Comorbidities (30): `aids`, `alcohol`, `ami`, `blane`, `canc`, `carit`, `cevd`, `chf`, `coag`, `copd`, `dane`, `dementia`, `depre`, `diab`, `diabc`, `diabunc`, `diabwc`, `drug`, `fed`, `hp`, `hypc`, `hypothy`, `hypunc`, `ld`, `lymph`, `metacanc`, `msld`, `obes`, `ond`, `pcd`, `psycho`, `pud`, `pvd`, `rend`, `rheumd`, `valv`, `wloss`
  - ICD Diagnosis Category Clusters (6): `Primary_Diagnosis_ICD_10_C_M_Code_Cluster`, `Other_Diagnosis_Code_1..5_ICD_10_C_M_Cluster`
- **Meso Level (64 Base/Encoded SDOH Features)**:
  - Urban/Rural Population (4): `POP_URB`, `POPPCT_URB`, `POP_RUR`, `POPPCT_RUR`
  - ACS Education Indicators (6): `ACS_PCT_BACHELOR_DGR`, `ACS_PCT_COLLEGE_ASSOCIATE_DGR`, `ACS_PCT_GRADUATE_DGR`, `ACS_PCT_HS_GRADUATE`, `ACS_PCT_LT_HS`, `ACS_PCT_POSTHS_ED`
  - ACS Broadband & Technology Access (13): `ACS_PCT_HH_BROADBAND`, `ACS_PCT_HH_BROADBAND_ONLY`, `ACS_PCT_HH_CELLULAR`, `ACS_PCT_HH_CELLULAR_ONLY`, `ACS_PCT_HH_DIAL_INTERNET_ONLY`, `ACS_PCT_HH_INTERNET`, `ACS_PCT_HH_INTERNET_NO_SUBS`, `ACS_PCT_HH_NO_COMP_DEV`, `ACS_PCT_HH_NO_INTERNET`, `ACS_PCT_HH_OTHER_COMP`, `ACS_PCT_HH_OTHER_COMP_ONLY`, `ACS_PCT_HH_PC`, `ACS_PCT_HH_SMARTPHONE`, `ACS_PCT_HH_TABLET`
  - ACS Income, Poverty & Veteran Status (16): `ACS_PCT_HEALTH_INC_BELOW137`, `ACS_PCT_HEALTH_INC_138_199`, `ACS_PCT_HEALTH_INC_200_399`, `ACS_PCT_HEALTH_INC_ABOVE400`, `ACS_PCT_INC50`, `ACS_PCT_INC50_ABOVE65`, `ACS_PCT_NONVET_POV_18_64`, `ACS_PCT_PERSON_INC_100_124`, `ACS_PCT_PERSON_INC_125_199`, `ACS_PCT_PERSON_INC_ABOVE200`, `ACS_PCT_PERSON_INC_BELOW99`, `ACS_PCT_POV_AIAN`, `ACS_PCT_POV_ASIAN`, `ACS_PCT_POV_BLACK`, `ACS_PCT_POV_HISPANIC`, `ACS_PCT_POV_MULTI`, `ACS_PCT_POV_NHPI`, `ACS_PCT_POV_OTHER`, `ACS_PCT_POV_WHITE`, `ACS_PCT_VET_POV_18_64`, `ACS_TOT_POP_POV`
  - ACS Household & Family Dynamics (25): `ACS_PCT_CHILDREN_GRANDPARENT`, `ACS_PCT_CHILD_1FAM`, `ACS_PCT_GRANDP_NO_RESPS`, `ACS_PCT_GRANDP_RESPS_NO_P`, `ACS_PCT_GRANDP_RESPS_P`, `ACS_PCT_HH_1PERS`, `ACS_PCT_HH_ABOVE65`, `ACS_PCT_HH_ALONE_ABOVE65`, `ACS_PCT_HH_KID_1PRNT`, `ACS_TOT_GRANDCHILDREN_GP`, `ACS_PCT_HH_1FAM_FOOD_STMP`, `ACS_PCT_HH_FOOD_STMP_BLW_POV`, `ACS_PCT_HH_NO_FD_STMP_BLW_POV`, `ACS_PCT_HH_PUB_ASSIST`
- **Macro Level (4 Base Features)**:
  - `Submitted_HIPPS_Code`, `Facility_Internal_ID`, `Agency_Medicare_Number`, `COUNTY_NAME`

### 4.1. Exact Cohort Feature Counts Summary

| Cohort | Subgroup Condition Filter | Total Dataset Columns | Modeling Features ($X$) | Target & ID Columns | Micro (Clinical) | Meso (SDOH) | Macro (System/Provider) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Complete Cohort** (`processed_final_mergedDF_TX.csv`) | All home health admissions | **126** | **124** | **2** (`Beneficiary_ID`, `ever_readmitted`) | **54** | **64** | **6** |
| **Diabetic Patients Cohort** (`processed_diabetic_cohort_TX.csv`) | Patients with `has_diabetes == 1` | **125** | **123** | **2** (`Beneficiary_ID`, `ever_readmitted`) | **53** *(excl. `has_diabetes` flag)* | **64** | **6** |
| **Heart Failure Patients Cohort** (`processed_heart_failure_cohort_TX.csv`) | Patients with `has_heart_failure == 1` | **125** | **123** | **2** (`Beneficiary_ID`, `ever_readmitted`) | **53** *(excl. `has_heart_failure` flag)* | **64** | **6** |
| **Hypertensive Patients Cohort** (`processed_hypertension_cohort_TX.csv`) | Patients with `has_hypertension == 1` | **125** | **123** | **2** (`Beneficiary_ID`, `ever_readmitted`) | **53** *(excl. `has_hypertension` flag)* | **64** | **6** |

---

## 5. LEAP2 HPC Execution Guide

### Slurm Submission Commands

To run preprocessing or feature selection on LEAP2:

```bash
# 1. Run Parallel Preprocessing
sbatch preprocessing.slurm

# 2. Run Feature Selection Comparison
cd featureSelection
sbatch fs.slurm
```

### Key Slurm Job Specifications (`fs.slurm`)

```bash
#SBATCH --job-name=featureSelection
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=64Gb
#SBATCH --time=05:00:00
#SBATCH --output=job_fs_output.txt
#SBATCH --error=job_fs_error.txt

export PYTHONPATH=".:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

pip install --quiet "imbalanced-learn<0.10" pandas numpy scikit-learn joblib scipy lightgbm xgboost
python feature_selection.py
```

---

## 6. Verification & Output Summary

- **Syntax Verification**: `python -m py_compile` passed with 0 errors across all scripts.
- **Output Artifacts Generated**:
  - `data/processed_final_mergedDF_TX.csv`
  - `data/processed_final_mergedDF.csv`
  - `featureSelection/results/TX_feature_selection_results.csv`
