#!/bin/bash
#SBATCH --job-name=08_parity_tx
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_08_parity_tx_%j.out
#SBATCH --error=job_08_parity_tx_%j.err

echo "================================================================="
echo "  STEP 8A: Unified Parity & Fairness Benchmark - TEXAS COHORT"
echo "  Evaluating Demographic Parity, Equalized Odds, and Calibration Gaps"
echo "  Running on Host: $SLURM_NODELIST"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Start Time: $(date)"
echo "================================================================="

ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR/parity" || exit 1
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
python -u run_unified_parity_comparison.py --cohort Texas 2>&1 | tee run_unified_parity_texas.log

STATUS=$?
conda deactivate 2>/dev/null || true

echo "================================================================="
if [ $STATUS -eq 0 ]; then
    echo "  STEP 8A TEXAS UNIFIED PARITY BENCHMARK COMPLETED SUCCESSFULLY AT $(date)"
else
    echo "  STEP 8A TEXAS UNIFIED PARITY BENCHMARK FAILED WITH EXIT CODE $STATUS AT $(date)"
fi
echo "================================================================="
exit $STATUS
