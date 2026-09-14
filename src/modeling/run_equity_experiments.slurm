#!/bin/bash
#SBATCH --job-name=oasis_equity_experiments
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_equity_%j.out
#SBATCH --error=job_equity_%j.err

echo ""
echo "Starting Algorithmic Equity & Bias Mitigation Pipeline at $(date)"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

# Change directory
cd "/home/m_e172/OASIS/OASIS_v2/data processing - newest/modeling" || exit 1
echo "Current working directory is $(pwd)"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

# Ensure required packages are installed
pip install --quiet transformers torch scikit-learn lightgbm xgboost catboost imbalanced-learn tabulate 2>/dev/null || true

# Set PYTHONPATH to include current and parent directories for clean imports
export PYTHONPATH=".:..:../featureSelection:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Active Python: $(which python)"
python -u -c "import sys; print('Python executable:', sys.executable)"

echo ""
echo "================================================================="
echo "  Executing Standalone Algorithmic Equity Evaluation Pipeline"
echo "================================================================="
python -u run_equity_experiments.py 2>&1 | tee run_equity.log
STATUS=${PIPESTATUS[0]}
conda deactivate 2>/dev/null || true
echo ""
echo "Equity pipeline finished with exit code $STATUS at: $(date)"
echo ""
