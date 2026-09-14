#!/bin/bash
#SBATCH --job-name=oasis_modeling
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_modeling_%j.out
#SBATCH --error=job_modeling_%j.err

echo ""
echo "Starting Model Pipeline at $(date)"
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

# Ensure required modeling packages are installed in spark_env
echo "Verifying / Installing PyTorch, Transformers, and ML packages..."
pip install --quiet transformers torch scikit-learn lightgbm xgboost catboost imbalanced-learn pyarrow fastparquet 2>/dev/null || true

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
echo "  STEP 1: Running Tree/Ensemble Modeling (run_modeling.py)..."
echo "================================================================="
python -u run_modeling.py 2>&1 | tee run_modeling.log
STATUS_M1=${PIPESTATUS[0]}
echo "run_modeling.py completed with exit code $STATUS_M1 at $(date)"

echo ""
echo "================================================================="
echo "  STEP 2: Running PyTorch Neural Modeling (run_neural_modeling.py)..."
echo "================================================================="
python -u run_neural_modeling.py 2>&1 | tee run_neural_modeling.log
STATUS_M2=${PIPESTATUS[0]}
echo "run_neural_modeling.py completed with exit code $STATUS_M2 at $(date)"

echo ""
echo "================================================================="
if [ $STATUS_M1 -eq 0 ] && [ $STATUS_M2 -eq 0 ]; then
    echo "  ALL MODELING EXPERIMENTS COMPLETED SUCCESSFULLY AT $(date)"
else
    echo "  MODELING EXPERIMENTS FINISHED WITH EXIT CODES: run_modeling=$STATUS_M1, run_neural_modeling=$STATUS_M2 AT $(date)"
fi
echo "================================================================="

conda deactivate 2>/dev/null || true
echo ""

