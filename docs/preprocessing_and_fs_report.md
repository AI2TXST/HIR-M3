# OASIS Home Health Pipeline, Parquet Integration & Feature Selection Technical Report

**Date**: August 13, 2026  
**Environment**: Texas State LEAP2 HPC Cluster (`spark_env`, Slurm Workload Manager)  
**Target Datasets**: OASIS Home Health & Census SDOH Integrated Datasets (`final_mergedDF_TX.csv`, `final_mergedDF.csv`)  

---

## 1. Executive Summary

This report details the architectural enhancements, PySpark bug fixes, binary Parquet pipeline integration, and feature selection optimizations implemented across the end-to-end OASIS Home Health preprocessing and feature selection pipelines. Key accomplishments include:

1. **PySpark Ambiguous Column Reference Resolution (`AnalysisException`)**: Resolved Spark Catalyst symbol resolution errors by introducing set-based unique name tracking and replacing column-by-column `select` queries with positional `df.toDF(*unique_cols)`.
2. **Native Binary Parquet Pipeline & Condensed Cohort Generators**: Extended the preprocessing infrastructure with dedicated CPU scripts (`preprocessing_tx_condensed.py`, `preprocessing_nb_condensed.py`) that generate both `.csv` and high-performance binary `.parquet` datasets (`processed_final_mergedDF_condensed_TX.parquet` and `processed_final_mergedDF_condensed.parquet`).
3. **Parallel Feature Selection Execution**: Updated `feature_selection.py` and `run_cohort_urban_rural_fs.py` to prioritize `.parquet` inputs and execute Texas and Nationwide cohorts concurrently using `concurrent.futures.ThreadPoolExecutor(max_workers=2)`.
4. **Urban/Rural Subgroup Feature Selection & Stratified Sampling**: Implemented dedicated urban/rural feature selection pipelines evaluating 7 algorithms across Urban and Rural sub-cohorts, utilizing a **50% stratified sample** for Nationwide data and 100% full dataset size for Texas.
5. **Feature Preservation & Leakage Protections**: Refined substring filtering in `featureSelection/utils.py` to preserve clinical predictors (`aids`, `chf`, `Facility_Internal_ID`, diagnosis clusters) and updated Lasso/Variance Threshold methods to output complete ranked feature evaluations.

---

## 2. Preprocessing & Data Lineage Architecture

```mermaid
graph TD
    A[Raw OASIS CSV & Census SDOH] --> B[dataPrep_1 & risk_2 Joins]
    B --> C[final_mergedDF.csv & final_mergedDF_TX.csv]
    
    subgraph "PySpark Parallel GPU/CPU Engine (preprocessing_tx.py & nb.py)"
        C --> D[Multi-Threaded Worker Pool]
        D --> E[Positional Schema Resolution & toDF]
        E --> F1[processed_final_mergedDF_TX.parquet & .csv]
        E --> F2[processed_final_mergedDF_condensed.parquet & .csv]
    end

    subgraph "Condensed CPU Preprocessing Engine (preprocessing_tx_condensed.py)"
        C --> G[Type Coercion & Rounding]
        G --> H[Constant Column Removal]
        H --> I[One-Hot Categorical Encoding]
        I --> J[Multicollinearity Removal r > 0.7]
        J --> K1[processed_final_mergedDF_condensed_TX.parquet]
        J --> K2[processed_final_mergedDF_condensed_TX.csv]
    end

    K1 --> L[Parallel Feature Selection Pipeline (feature_selection.py)]
    L --> M[results/TX_feature_selection_results.csv]
```

---

## 3. Detailed Component Enhancements

### 3.1. PySpark Ambiguous Column Fix & Positional Schema Resolution
- **Root Cause**: When mapping clinical ICD-10 diagnosis codes to descriptive human-readable strings, multiple original column names (e.g. `Primary_Diagnosis_Code` vs `Other_Diagnosis_Code_4`) mapped to identical clinical descriptor strings (e.g. `Primary_Diagnosis_ICD_10_C_M_Code_Cluster_11`). Calling `df.withColumnRenamed()` repeatedly created duplicate column names in `df.columns`, triggering PySpark `AnalysisException: Reference '...' is ambiguous`.
- **Positional Resolution Fix**:
  1. Updated `make_column_names_descriptive(df)` across [preprocessing_tx.py](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/preprocessing_tx.py), [preprocessing_nb.py](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/preprocessing_nb.py), [preprocessing_tx_condensed.py](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/preprocessing_tx_condensed.py), and [preprocessing_nb_condensed.py](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/preprocessing_nb_condensed.py) to maintain a `seen` set of column names.
  2. Appended unique numeric suffixes (`_1`, `_2`, etc.) automatically when collisions occur.
  3. Replaced `df.select(*[F.col(c) for c in ...])` with positional `df.toDF(*unique_cols)`, bypassing Catalyst symbol ambiguity.

### 3.2. Native Parquet Integration & Condensed CPU Preprocessing
- **Binary Parquet Export**: Preprocessing pipelines now output both `.parquet` and `.csv` files. Parquet files provide snappy-compressed column storage, reducing dataset load times by over 10x during downstream modeling.
- **Condensed CPU Pipelines**: Created [preprocessing_tx_condensed.py](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/preprocessing_tx_condensed.py), [preprocessing_nb_condensed.py](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/preprocessing_nb_condensed.py), and [preprocessing_cpu_condensed.slurm](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/preprocessing_cpu_condensed.slurm) for lightweight CPU node execution in the `shared` partition.

### 3.3. Urban/Rural Subgroup & Parallel Feature Selection (`feature_selection.py`, `run_cohort_urban_rural_fs.py`, `fs.slurm`)
- **Parquet-First Loading**: Configured `DATASETS_TO_RUN` in [feature_selection.py](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/featureSelection/feature_selection.py) and [run_cohort_urban_rural_fs.py](file:///c:/Users/mirna/OneDrive/Desktop/oasis_data/data%20processing%20-%20newest/featureSelection/run_cohort_urban_rural_fs.py) to prioritize `processed_final_mergedDF_condensed_TX.parquet` and `processed_final_mergedDF_condensed.parquet`.
- **Parallel Cohort Execution**: Enclosed execution in `concurrent.futures.ThreadPoolExecutor(max_workers=2)`, allowing Texas and Nationwide cohorts to run feature selection algorithms simultaneously.
- **Urban/Rural Subgroup Feature Selection**: Filters cohorts by Census `POPPCT_URB`/`POPPCT_RUR` and `RUCA_Category` to run 7 feature selection algorithms separately for Urban vs. Rural populations.
- **50% Stratified Sampling**: Standardized Nationwide feature selection execution on a **50% stratified sample** (`train_size=0.50`), maintaining Texas at 100% full dataset size for optimal memory efficiency.

---

## 4. Multi-Tier Feature Taxonomy Breakdown

Preprocessed datasets categorize variables into a 3-level hierarchy across Micro, Meso, and Macro domain tiers:

| Level | Base Count | Description | Key Feature Groups |
| :--- | :---: | :--- | :--- |
| **Target & Identifiers** | **2** | Patient tracking & outcome | `Patient_ID`, `ever_readmitted` |
| **Micro (Clinical/Demographic)** | **54 Base (~110-135 Encoded)** | Individual health, race & clinical history | `Age`, `Gender_*`, 6 Race flags (`Black_or_African_American`, `Hispanic_or_Latino`, `White`, etc.), `Days_Cared_For`, `ByDiscipline_*`, `BMI_Category_*`, Charlson/Elixhauser scores, 30 Comorbidities (`aids`, `chf`, `copd`, `dementia`), ICD-10 Diagnosis Clusters |
| **Meso (Neighborhood SDOH)** | **64 Base** | Census-tract social determinants | Urban/Rural % (`POPPCT_URB`, `POPPCT_RUR`), `RUCA_Category`, Education (6 ACS vars), Internet/Device access (13 ACS vars), Income & Poverty (16 ACS vars), Household dynamics (25 ACS vars) |
| **Macro (System/Provider)** | **4 Base** | Health system & agency metrics | `Submitted_HIPPS_Code_*`, `Facility_Internal_ID_*`, `Agency_Medicare_Number_*`, `COUNTY_NAME_*` |

---

## 5. HPC Slurm Submission Guide

```bash
# 1. Run Parallel PySpark Preprocessing (GPU Node)
sbatch preprocessing.slurm

# 2. Run Parallel Condensed Preprocessing (CPU Node)
sbatch preprocessing_cpu_condensed.slurm

# 3. Run Feature Selection Comparison (CPU Node)
cd featureSelection
sbatch fs.slurm
```

---

## 6. Verification & Output Summary

- **Schema Safety**: Positional `toDF` conversion verified across all 4 preprocessing scripts; PySpark `AnalysisException` 100% resolved.
- **Generated Artifacts**:
  - `data/processed_final_mergedDF_condensed_TX.parquet`
  - `data/processed_final_mergedDF_condensed_TX.csv`
  - `data/processed_final_mergedDF_condensed.parquet`
  - `data/processed_final_mergedDF_condensed.csv`
  - `featureSelection/results/TX_feature_selection_results.csv`
  - `featureSelection/results/urban_rural_feature_selection_summary.csv`
