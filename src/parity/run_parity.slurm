#!/bin/bash
#SBATCH --job-name=act_parity_v2
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_act_parity_%j.out
#SBATCH --error=job_act_parity_%j.err

echo ""
echo "================================================================="
echo "  ACT-PARITY V2: TOKENIZED MULTI-TIER PARALLEL EVALUATION PIPELINE"
echo "  Running on Host: $SLURM_NODELIST"
echo "  Job ID: $SLURM_JOB_ID"
echo "================================================================="

ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR/parity" || exit 1
echo "Current working directory: $(pwd)"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

# Ensure pyarrow, fastparquet, torch, scikit-learn are installed
pip install --quiet pyarrow fastparquet torch scikit-learn lightgbm xgboost catboost imbalanced-learn 2>/dev/null || true

export PYTHONPATH=".:..:$ROOT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Active Python: $(which python)"

echo ""
echo "Launching ACT-Parity v2 Factorial Ablation Pipeline (run_experiments.py)..."
python -u run_experiments.py

STATUS=$?
conda deactivate 2>/dev/null || true

echo ""
echo "================================================================="
if [ $STATUS -eq 0 ]; then
    echo "  ACT-PARITY V2 PIPELINE COMPLETED SUCCESSFULLY AT $(date)"
else
    echo "  ACT-PARITY V2 PIPELINE FAILED WITH EXIT CODE $STATUS AT $(date)"
fi
echo "================================================================="
echo ""
