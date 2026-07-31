#!/bin/bash
#SBATCH --job-name=tx_modeling_pipeline
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_tx_pipeline_%j.out
#SBATCH --error=job_tx_pipeline_%j.err

echo ""
echo "================================================================="
echo "  TEXAS COHORT COMPLETE MODELING PIPELINE STARTED AT $(date)"
echo "  Running on Host: $SLURM_NODELIST"
echo "  Job ID: $SLURM_JOB_ID"
echo "================================================================="

# Set root working directory to modeling/tx
ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR/modeling/tx" || exit 1
echo "Current working directory: $(pwd)"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

# Configure PYTHONPATH and Thread Limits
export PYTHONPATH=".:..:$ROOT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Active Python: $(which python)"

# -----------------------------------------------------------------
# STEP 1: TEXAS COHORT BASELINE & NEURAL MODELING
# -----------------------------------------------------------------
echo ""
echo "-----------------------------------------------------------------"
echo "  STEP 1: TEXAS COHORT BASELINE & NEURAL MODELING"
echo "-----------------------------------------------------------------"
python -u run_tx_modeling.py
STEP1_STATUS=$?

if [ $STEP1_STATUS -ne 0 ]; then
    echo "ERROR: Step 1 (Texas Modeling) failed with exit code $STEP1_STATUS at $(date)."
    exit $STEP1_STATUS
fi
echo "Step 1 (Texas Modeling) saved and finished successfully at $(date)."

# -----------------------------------------------------------------
# STEP 2: TEXAS URBAN VS. RURAL MODELING
# -----------------------------------------------------------------
echo ""
echo "-----------------------------------------------------------------"
echo "  STEP 2: TEXAS URBAN VS. RURAL MODELING"
echo "-----------------------------------------------------------------"
python -u run_tx_urban_rural.py
STEP2_STATUS=$?

if [ $STEP2_STATUS -ne 0 ]; then
    echo "ERROR: Step 2 (Texas Urban vs Rural) failed with exit code $STEP2_STATUS at $(date)."
    exit $STEP2_STATUS
fi
echo "Step 2 (Texas Urban vs Rural) saved and finished successfully at $(date)."

# -----------------------------------------------------------------
# STEP 3: TEXAS ENSEMBLE RATIO EXPLORATION
# -----------------------------------------------------------------
echo ""
echo "-----------------------------------------------------------------"
echo "  STEP 3: TEXAS ENSEMBLE RATIO EXPLORATION"
echo "-----------------------------------------------------------------"
python -u run_tx_ensemble_ratios.py
STEP3_STATUS=$?

if [ $STEP3_STATUS -ne 0 ]; then
    echo "ERROR: Step 3 (Texas Ensemble Ratios) failed with exit code $STEP3_STATUS at $(date)."
    exit $STEP3_STATUS
fi
echo "Step 3 (Texas Ensemble Ratios) saved and finished successfully at $(date)."

conda deactivate 2>/dev/null || true
echo ""
echo "================================================================="
echo "  ALL 3 TEXAS PIPELINE STEPS COMPLETED & SAVED SUCCESSFULLY AT $(date)"
echo "================================================================="
