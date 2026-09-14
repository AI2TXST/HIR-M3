#!/bin/bash
# ==============================================================================
# Master SLURM Pipeline Submission Script for Leap Cluster
# Submits all primary analyses in strict dependency order with --dependency=afterok
# ==============================================================================

echo "================================================================="
echo "  SUBMITTING FULL OASIS PRIMARY ANALYSIS PIPELINE TO SLURM"
echo "  Start Time: $(date)"
echo "================================================================="

ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR" || exit 1

# Step 1: Clean Data Preparation (Leak-Free Datasets)
JOB_01=$(sbatch --parsable 01_build_clean_datasets.slurm)
echo "[1/9] Submitted Step 1 (Data Prep): Job ID $JOB_01"

# Step 2: Feature Selection (Depends on Step 1)
JOB_02=$(sbatch --parsable --dependency=afterok:$JOB_01 02_feature_selection.slurm)
echo "[2/9] Submitted Step 2 (Feature Selection): Job ID $JOB_02 (after $JOB_01)"

# Step 3: Baseline ML Modeling (Depends on Step 1)
JOB_03=$(sbatch --parsable --dependency=afterok:$JOB_01 03_primary_modeling.slurm)
echo "[3/9] Submitted Step 3 (Baseline ML Modeling): Job ID $JOB_03 (after $JOB_01)"

# Step 4: Neural Modeling (Depends on Step 1)
JOB_04=$(sbatch --parsable --dependency=afterok:$JOB_01 04_neural_modeling.slurm)
echo "[4/9] Submitted Step 4 (Neural Modeling): Job ID $JOB_04 (after $JOB_01)"

# Step 5: Ensemble Ratios (Depends on Steps 3 & 4)
JOB_05=$(sbatch --parsable --dependency=afterok:$JOB_03:$JOB_04 05_ensemble_ratios.slurm)
echo "[5/9] Submitted Step 5 (Ensemble Ratios): Job ID $JOB_05 (after $JOB_03, $JOB_04)"

# Step 6: Urban vs. Rural Stratified Modeling (Depends on Step 1)
JOB_06=$(sbatch --parsable --dependency=afterok:$JOB_01 06_urban_rural_modeling.slurm)
echo "[6/9] Submitted Step 6 (Urban/Rural Modeling): Job ID $JOB_06 (after $JOB_01)"

# Step 7: Condition Subgroups (Depends on Step 1)
JOB_07=$(sbatch --parsable --dependency=afterok:$JOB_01 07_condition_subgroups.slurm)
echo "[7/9] Submitted Step 7 (Condition Subgroups): Job ID $JOB_07 (after $JOB_01)"

# Step 8: Unified Parity Benchmark (Depends on Step 1)
JOB_08=$(sbatch --parsable --dependency=afterok:$JOB_01 08_unified_parity.slurm)
echo "[8/9] Submitted Step 8 (Unified Parity Benchmark): Job ID $JOB_08 (after $JOB_01)"

# Step 9: Equity & Bias Experiments (Depends on Step 1)
JOB_09=$(sbatch --parsable --dependency=afterok:$JOB_01 09_equity_experiments.slurm)
echo "[9/10] Submitted Step 9 (Equity Experiments): Job ID $JOB_09 (after $JOB_01)"

# Step 10: Reviewer Evaluation Suite Runs 1-4 (Depends on Steps 5 & 8)
JOB_10=$(sbatch --parsable --dependency=afterok:$JOB_05:$JOB_08 10_reviewer_experiments.slurm)
echo "[10/10] Submitted Step 10 (Reviewer Experiments Runs 1-4): Job ID $JOB_10 (after $JOB_05, $JOB_08)"

echo "================================================================="
echo "  ALL 10 JOBS SUCCESSFULLY SUBMITTED TO SLURM!"
echo "  Check queue status with: squeue -u \$USER"
echo "================================================================="

