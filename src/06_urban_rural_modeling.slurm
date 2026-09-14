#!/bin/bash
#SBATCH --job-name=06_urban_rural
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_06_urban_rural_%j.out
#SBATCH --error=job_06_urban_rural_%j.err

echo "================================================================="
echo "  STEP 6: Urban vs. Rural Stratified Modeling"
echo "  Running on Host: $SLURM_NODELIST"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Start Time: $(date)"
echo "================================================================="

ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR/modeling" || exit 1
echo "Current working directory: $(pwd)"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

pip install --quiet torch scikit-learn lightgbm xgboost catboost imbalanced-learn 2>/dev/null || true

export PYTHONPATH=".:..:$ROOT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Active Python: $(which python)"

echo ""
echo "--- Running ML Baselines for Urban & Rural Subgroups ---"
python -u run_urban_rural_modeling.py 2>&1 | tee run_urban_rural_modeling.log
STATUS_UR1=${PIPESTATUS[0]}

echo ""
echo "--- Running Neural Architectures for Urban & Rural Subgroups ---"
python -u run_urban_rural_neural_modeling.py 2>&1 | tee run_urban_rural_neural_modeling.log
STATUS_UR2=${PIPESTATUS[0]}

conda deactivate 2>/dev/null || true

echo "================================================================="
if [ $STATUS_UR1 -eq 0 ] && [ $STATUS_UR2 -eq 0 ]; then
    echo "  STEP 6 URBAN/RURAL MODELING COMPLETED SUCCESSFULLY AT $(date)"
    FINAL_STATUS=0
else
    echo "  STEP 6 URBAN/RURAL MODELING ENCOUNTERED ERRORS (ML: $STATUS_UR1, Neural: $STATUS_UR2) AT $(date)"
    FINAL_STATUS=1
fi
echo "================================================================="
exit $FINAL_STATUS
