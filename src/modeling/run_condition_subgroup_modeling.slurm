#!/bin/bash
#SBATCH --job-name=condition_subgroups_cpu
#SBATCH --output=condition_subgroups_%j.out
#SBATCH --error=condition_subgroups_%j.err
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_condition_subgroups_%j.out
#SBATCH --error=job_condition_subgroups_%j.err

echo ""
echo "Starting Condition Subgroups Pipeline at $(date)"
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
echo "  Running All 10 Models in Parallel on CPU (8 Worker Threads)"
echo "-----------------------------------------------------------------"
python -u run_condition_subgroup_modeling.py --workers 8
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Parallel CPU condition subgroup modeling failed with exit code $STATUS at $(date)."
    exit $STATUS
fi

conda deactivate 2>/dev/null || true
echo ""
echo "================================================================="
echo "  PARALLEL CPU CONDITION SUBGROUP PIPELINE FINISHED AT $(date)"
echo "================================================================="
