#!/bin/bash
#SBATCH --job-name=05_ens_nat
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_05_ens_nat_%j.out
#SBATCH --error=job_05_ens_nat_%j.err

echo "================================================================="
echo "  STEP 5B: Ensemble Ratios & Optimization - NATIONWIDE COHORT"
echo "  Evaluating GBDT + HIR-M3 Blending Ratios (100:0 down to 0:100)"
echo "  Running on Host: $SLURM_NODELIST"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Start Time: $(date)"
echo "================================================================="

ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR/modeling" || exit 1
echo "Current working directory: $(pwd)"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

pip install --quiet torch scikit-learn lightgbm xgboost catboost imbalanced-learn 2>/dev/null || true

export PYTHONPATH=".:..:$ROOT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Active Python: $(which python)"

echo ""
echo "-----------------------------------------------------------------"
echo "  1. Running Baseline ML vs. HIR-M3 Ratio Exploration (Nationwide)"
echo "-----------------------------------------------------------------"
python -u explore_ensemble_ratios.py --cohort Nationwide 2>&1 | tee explore_ensemble_ratios_nationwide.log
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Baseline ensemble ratio experiment for Nationwide failed with exit code $STATUS at $(date)."
    exit $STATUS
fi

echo ""
echo "-----------------------------------------------------------------"
echo "  2. Running Neural Models vs. HIR-M3 Ratio Exploration (Nationwide)"
echo "-----------------------------------------------------------------"
python -u explore_neural_ensemble_ratios.py --cohort Nationwide 2>&1 | tee explore_neural_ensemble_ratios_nationwide.log
NEURAL_STATUS=$?

conda deactivate 2>/dev/null || true

echo "================================================================="
if [ $NEURAL_STATUS -eq 0 ]; then
    echo "  STEP 5B NATIONWIDE ENSEMBLE RATIOS COMPLETED SUCCESSFULLY AT $(date)"
else
    echo "  STEP 5B NATIONWIDE ENSEMBLE RATIOS FAILED WITH EXIT CODE $NEURAL_STATUS AT $(date)"
fi
echo "================================================================="
exit $NEURAL_STATUS
