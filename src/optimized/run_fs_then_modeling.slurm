#!/bin/bash
#SBATCH --job-name=oasis_fs_then_modeling
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_fs_then_modeling_%j.out
#SBATCH --error=job_fs_then_modeling_%j.err

echo ""
echo "================================================================="
echo "  OASIS FEATURE SELECTION -> MODELING PIPELINE STARTED AT $(date)"
echo "  Running on Host: $SLURM_NODELIST"
echo "  Job ID: $SLURM_JOB_ID"
echo "================================================================="

# Set root directory
ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR" || exit 1
echo "Current working directory: $(pwd)"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

# Configure PYTHONPATH and Thread Limits
export PYTHONPATH=".:featureSelection:modeling:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Active Python: $(which python)"
python -u -c "import sys; print('Python executable:', sys.executable)"

# -----------------------------------------------------------------
# STAGE 1: FEATURE SELECTION
# -----------------------------------------------------------------
#echo ""
#echo "-----------------------------------------------------------------"
#echo "  STAGE 1: FEATURE SELECTION (Nationwide Condensed)"
#echo "-----------------------------------------------------------------"
#cd "$ROOT_DIR/featureSelection" || exit 1
#python -u feature_selection.py
#FS_STATUS=$?

#if [ $FS_STATUS -ne 0 ]; then
#    echo "ERROR: Feature selection failed with exit code $FS_STATUS at $(date). Aborting modeling."
#    exit $FS_STATUS
#fi
#echo "Stage 1 Feature Selection completed successfully at $(date)."

# -----------------------------------------------------------------
# STAGE 2: BASELINE ML & PYTORCH NEURAL MODELING
# -----------------------------------------------------------------
echo ""
echo "-----------------------------------------------------------------"
echo "  STAGE 2A: BASELINE MACHINE LEARNING MODELING"
echo "-----------------------------------------------------------------"
cd "$ROOT_DIR/modeling" || exit 1
python -u run_modeling.py
MODELING_STATUS=$?

if [ $MODELING_STATUS -ne 0 ]; then
    echo "ERROR: Baseline modeling failed with exit code $MODELING_STATUS at $(date)."
    exit $MODELING_STATUS
fi

echo ""
echo "-----------------------------------------------------------------"
echo "  STAGE 2B: PYTORCH NEURAL & TRANSFORMER MODELING"
echo "-----------------------------------------------------------------"
python -u run_neural_modeling.py
NEURAL_STATUS=$?

if [ $NEURAL_STATUS -ne 0 ]; then
    echo "ERROR: Neural modeling failed with exit code $NEURAL_STATUS at $(date)."
    exit $NEURAL_STATUS
fi
echo "Stage 2 Baseline & Neural Modeling completed successfully at $(date)."

conda deactivate 2>/dev/null || true
echo ""
echo "================================================================="
echo "  PIPELINE FINISHED SUCCESSFULLY AT $(date)"
echo "================================================================="
