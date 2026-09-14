#!/bin/bash
#SBATCH --job-name=parity_subgroups
#SBATCH --output=job_parity_subgroups_%j.out
#SBATCH --error=job_parity_subgroups_%j.err
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu

echo ""
echo "================================================================="
echo "  STARTING NATIONWIDE VS TEXAS PARITY SUBGROUPS PIPELINE AT $(date)"
echo "  Running on Host: $SLURM_NODELIST"
echo "  Job ID: $SLURM_JOB_ID"
echo "================================================================="

# Change working directory
ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR/parity" || exit 1
echo "Current working directory: $(pwd)"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

# Verify / Install PyTorch, Transformers, and ML packages
echo "Verifying PyTorch, Transformers, and ML packages..."
pip install --quiet transformers torch scikit-learn lightgbm xgboost catboost imbalanced-learn pyarrow fastparquet 2>/dev/null || true

# Set PYTHONPATH for clean imports
export PYTHONPATH=".:..:../modeling:../featureSelection:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
export MKL_CBWR=COMPATIBLE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Active Python: $(which python)"
python -u -c "import sys; print('Python executable:', sys.executable)"

echo ""
echo "-----------------------------------------------------------------"
echo "  Executing Parity Experiments for Nationwide & Texas Cohorts"
echo "-----------------------------------------------------------------"
python -u run_parity_subgroups.py --workers 8
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Parity subgroup pipeline failed with exit code $STATUS at $(date)."
    exit $STATUS
fi

conda deactivate 2>/dev/null || true
echo ""
echo "================================================================="
echo "  PARITY SUBGROUPS PIPELINE FINISHED AT $(date)"
echo "================================================================="
