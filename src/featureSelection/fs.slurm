#!/bin/bash
#SBATCH --job-name=featureSelection
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64Gb
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_fs_%j.out
#SBATCH --error=job_fs_%j.err

echo ""
echo "Starting at $(date)"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

# path to the correct directory
cd "/home/m_e172/OASIS/OASIS_v2/data processing - newest/featureSelection" || exit 1
echo "Current working directory is $(pwd)"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

# Ensure pyarrow, fastparquet, and ML packages are installed
pip install --quiet pyarrow fastparquet lightgbm xgboost catboost scikit-learn imbalanced-learn 2>/dev/null || true

# Set PYTHONPATH and Thread counts for parallel feature selection
export PYTHONPATH=".:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Active Python: $(which python)"
python -u -c "import sys; print('Python executable:', sys.executable)"

echo "Launching feature_selection.py (unbuffered logs)..."
python -u feature_selection.py
STATUS_FS1=$?

if [ $STATUS_FS1 -ne 0 ]; then
    echo "ERROR: feature_selection.py failed with exit code $STATUS_FS1 at $(date)"
    exit $STATUS_FS1
fi

echo "Results from feature_selection.py saved. Flushing output buffers to disk..."
sync

echo "Launching run_cohort_urban_rural_fs.py..."
python -u run_cohort_urban_rural_fs.py
STATUS=$?
conda deactivate 2>/dev/null || true
echo ""
echo "Program finished with exit code $STATUS at: $(date)"
echo ""

