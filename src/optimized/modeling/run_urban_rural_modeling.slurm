#!/bin/bash
#SBATCH --job-name=urban_rural_modeling
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_urban_rural_%j.out
#SBATCH --error=job_urban_rural_%j.err

echo ""
echo "================================================================="
echo "  URBAN VS RURAL MODELING PIPELINE STARTED AT $(date)"
echo "  Running on Host: $SLURM_NODELIST"
echo "  Job ID: $SLURM_JOB_ID"
echo "================================================================="

# Set working directory
ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR/modeling" || exit 1
echo "Current working directory: $(pwd)"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

echo "Active Python: $(which python)"

echo ""
echo "-----------------------------------------------------------------"
echo "  Running Urban vs. Rural Baseline Machine Learning Modeling"
echo "-----------------------------------------------------------------"
python -u run_urban_rural_modeling.py
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Urban vs Rural baseline modeling failed with exit code $STATUS at $(date)."
    exit $STATUS
fi

echo ""
echo "-----------------------------------------------------------------"
echo "  Running Urban vs. Rural PyTorch Neural & Transformer Modeling"
echo "-----------------------------------------------------------------"
python -u run_urban_rural_neural_modeling.py
NEURAL_STATUS=$?

if [ $NEURAL_STATUS -ne 0 ]; then
    echo "ERROR: Urban vs Rural neural modeling failed with exit code $NEURAL_STATUS at $(date)."
    exit $NEURAL_STATUS
fi

conda deactivate 2>/dev/null || true
echo ""
echo "================================================================="
echo "  PIPELINE FINISHED SUCCESSFULLY AT $(date)"
echo "================================================================="
