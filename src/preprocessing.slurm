#!/bin/bash
#SBATCH --job-name=preprocessing_encoding
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_pre_output.txt
#SBATCH --error=job_pre_error.txt

echo ""
echo "================================================================="
echo "  TEXAS & NATIONWIDE PREPROCESSING STARTED AT $(date)"
echo "  Running on Host: $SLURM_NODELIST"
echo "  Job ID: $SLURM_JOB_ID"
echo "================================================================="

# Set working directory
ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR" || exit 1
echo "Current working directory is $(pwd)"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

# Ensure HuggingFace transformers, torch, and scikit-learn are installed in spark_env
echo "Verifying / Installing HuggingFace transformers & PyTorch..."
pip install --quiet transformers torch scikit-learn 2>/dev/null || true

# Set PYTHONPATH and Threading for PySpark & Transformers
export PYTHONPATH=".:$ROOT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

echo "Active Python: $(which python)"
python -u -c "import sys, torch; print('Python executable:', sys.executable); print('CUDA Available:', torch.cuda.is_available())"
nvidia-smi 2>/dev/null || true

echo "Starting TX and NB preprocessing in parallel..."
python -u preprocessing_tx.py > tx_preprocessing.log 2>&1 &
PID_TX=$!

python -u preprocessing_nb.py > nb_preprocessing.log 2>&1 &
PID_NB=$!

wait $PID_TX
STATUS_TX=$?

wait $PID_NB
STATUS_NB=$?

echo ""
echo "================================================================="
if [ $STATUS_TX -eq 0 ] && [ $STATUS_NB -eq 0 ]; then
    echo "  PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY AT $(date)"
else
    echo "  PREPROCESSING PIPELINE COMPLETED WITH ISSUES AT $(date)"
    echo "  TX Exit Code: $STATUS_TX | NB Exit Code: $STATUS_NB"
fi
echo "================================================================="

conda deactivate 2>/dev/null || true
echo ""
