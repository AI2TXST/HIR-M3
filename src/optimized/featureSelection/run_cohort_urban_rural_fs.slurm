#!/bin/bash
#SBATCH --job-name=urban_rural_fs
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_urban_rural_fs_%j.out
#SBATCH --error=job_urban_rural_fs_%j.err

echo ""
echo "================================================================="
echo "  TEXAS & NATIONWIDE URBAN/RURAL FEATURE SELECTION STARTED AT $(date)"
echo "  Running on Host: $SLURM_NODELIST"
echo "  Job ID: $SLURM_JOB_ID"
echo "================================================================="

# Set root directory to featureSelection
ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR/featureSelection" || exit 1
echo "Current working directory: $(pwd)"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

# Set PYTHONPATH and Thread Limits
export PYTHONPATH=".:..:$ROOT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Active Python: $(which python)"
python -u -c "import sys; print('Python executable:', sys.executable)"

echo ""
echo "Launching run_cohort_urban_rural_fs.py..."
python -u run_cohort_urban_rural_fs.py

STATUS=$?
conda deactivate 2>/dev/null || true

echo ""
echo "================================================================="
if [ $STATUS -eq 0 ]; then
    echo "  FEATURE SELECTION PIPELINE COMPLETED SUCCESSFULLY AT $(date)"
else
    echo "  FEATURE SELECTION PIPELINE FAILED WITH EXIT CODE $STATUS AT $(date)"
fi
echo "================================================================="
