#!/bin/bash
#SBATCH --job-name=04_neural_models
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_04_neural_%j.out
#SBATCH --error=job_04_neural_%j.err

echo "================================================================="
echo "  STEP 4: Neural Architectures (run_neural_modeling.py)"
echo "  Evaluating HIR-M3, MLP, Standard Transformer, SAINT with 1000-fold Bootstrap CIs"
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
python -u run_neural_modeling.py 2>&1 | tee run_neural_modeling.log

STATUS=$?
conda deactivate 2>/dev/null || true

echo "================================================================="
if [ $STATUS -eq 0 ]; then
    echo "  STEP 4 NEURAL MODELING COMPLETED SUCCESSFULLY AT $(date)"
else
    echo "  STEP 4 NEURAL MODELING FAILED WITH EXIT CODE $STATUS AT $(date)"
fi
echo "================================================================="
exit $STATUS
