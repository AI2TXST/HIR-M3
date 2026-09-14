#!/bin/bash
#SBATCH --job-name=08_nb_bench
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_08_nb_bench_%j.out
#SBATCH --error=job_08_nb_bench_%j.err

echo "================================================================="
echo "  NATIONWIDE BENCHMARK (50% Sample): Baseline -> Neural -> HIR-M3 -> ACT-Parity -> TabPFN/TabICL/TabFM"
echo "  Evaluating Discrimination, Calibration, Worst-Group FNR & Equity"
echo "  Running on Host: $SLURM_NODELIST"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Start Time: $(date)"
echo "================================================================="

ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
cd "$ROOT_DIR/parity" || exit 1
echo "Current working directory: $(pwd)"

# Robust Conda Environment Initialization (tabular_fm)
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate tabular_fm 2>/dev/null || source activate tabular_fm 2>/dev/null || conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

pip install --quiet torch scikit-learn lightgbm xgboost catboost imbalanced-learn tabpfn 2>/dev/null || true

export PYTHONPATH=".:..:$ROOT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Pre-cache TabPFN pre-trained weights (~100MB checkpoint)
python -c "import tabpfn; from tabpfn import TabPFNClassifier; m = TabPFNClassifier(device='cpu', N_ensemble_configurations=1)" 2>/dev/null || true

python compare_all_models_nationwide.py --sample_frac 0.50

echo "================================================================="
echo "  Nationwide Benchmark Completed: $(date)"
echo "================================================================="
