#!/bin/bash
#SBATCH --job-name=oasis_full_pipeline
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64Gb
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_full_pipeline_output.txt
#SBATCH --error=job_full_pipeline_error.txt

echo ""
echo "================================================================="
echo "  OASIS END-TO-END PIPELINE STARTED AT $(date)"
echo "  Running on Host: $SLURM_NODELIST"
echo "================================================================="

# Set root directory
ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR"
echo "Current working directory: $(pwd)"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
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
# STAGE 1: CONDENSED PYSPARK PREPROCESSING (Texas & Nationwide)
# -----------------------------------------------------------------
#echo ""
#echo "-----------------------------------------------------------------"
#echo "  STAGE 1: CONDENSED PYSPARK PREPROCESSING"
#echo "-----------------------------------------------------------------"
#python -u preprocessing_tx_condensed.py > log_tx_condensed.out 2>&1 &
#python -u preprocessing_nb_condensed.py > log_nb_condensed.out 2>&1 &
#wait
#echo "Stage 1 Preprocessing completed successfully at $(date)."

# -----------------------------------------------------------------
# STAGE 2: MULTI-COHORT FEATURE SELECTION
# -----------------------------------------------------------------
echo ""
echo "-----------------------------------------------------------------"
echo "  STAGE 2: MULTI-COHORT FEATURE SELECTION"
echo "-----------------------------------------------------------------"
cd "$ROOT_DIR/featureSelection"
python -u feature_selection.py
echo "Stage 2 Feature Selection completed successfully at $(date)."

# -----------------------------------------------------------------
# STAGE 3: UNIFIED MACHINE LEARNING & NEURAL MODELING
# -----------------------------------------------------------------
echo ""
echo "-----------------------------------------------------------------"
echo "  STAGE 3: UNIFIED MACHINE LEARNING & NEURAL MODELING"
echo "-----------------------------------------------------------------"
cd "$ROOT_DIR/modeling"
python -u run_modeling.py
echo "Stage 3 Modeling completed successfully at $(date)."

conda deactivate
echo ""
echo "================================================================="
echo "  FULL PIPELINE FINISHED WITH EXIT CODE $? AT $(date)"
echo "================================================================="
