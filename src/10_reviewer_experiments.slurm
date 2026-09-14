#!/bin/bash
#SBATCH --job-name=10_reviewer_evals
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_10_reviewer_evals_%j.out
#SBATCH --error=job_10_reviewer_evals_%j.err

echo "================================================================="
echo "  STEP 10: Comprehensive Reviewer Evaluation Pipelines (Runs 1-4)"
echo "  Run 1: Cross-Cohort Transportability (Nationwide -> Texas)"
echo "  Run 2: Alert Volume Capacity Audit (Top 5%, 10%, 15%)"
echo "  Run 3: Decision Curve Analysis (DCA Net Benefit Curves)"
echo "  Run 4: 10-Decile Calibration Reliability & Recalibration Curves"
echo "  Running on Host: $SLURM_NODELIST"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Start Time: $(date)"
echo "================================================================="

ROOT_DIR="/home/m_e172/OASIS/OASIS_v2/data processing - newest"
if [ -d "$ROOT_DIR" ]; then
    cd "$ROOT_DIR" || exit 1
else
    # Fallback to local execution directory if running elsewhere
    cd "$(dirname "$0")" || exit 1
fi
echo "Current working directory: $(pwd)"

# Create necessary output directories
mkdir -p results/reviewer_experiments figures

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

pip install --quiet numpy scipy pandas scikit-learn matplotlib 2>/dev/null || true

export PYTHONPATH=".:scripts:$ROOT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Active Python: $(which python)"
echo "-----------------------------------------------------------------"

# ==============================================================================
# RUN 1: Cross-Cohort External Transportability Evaluation
# ==============================================================================
echo ""
echo ">>> [RUN 1/4] Executing scripts/eval_transport.py..."
python -u scripts/eval_transport.py 2>&1 | tee results/reviewer_experiments/run1_transport.log
RUN1_STATUS=$?

if [ $RUN1_STATUS -ne 0 ]; then
    echo "ERROR: Run 1 (Transportability) failed with exit code $RUN1_STATUS"
fi

# ==============================================================================
# RUN 2: Alert Volume Capacity Audit
# ==============================================================================
echo ""
echo ">>> [RUN 2/4] Executing scripts/eval_capacity.py..."
python -u scripts/eval_capacity.py 2>&1 | tee results/reviewer_experiments/run2_capacity.log
RUN2_STATUS=$?

if [ $RUN2_STATUS -ne 0 ]; then
    echo "ERROR: Run 2 (Capacity Audit) failed with exit code $RUN2_STATUS"
fi

# ==============================================================================
# RUN 3: Decision Curve Analysis (DCA Net Benefit Curves)
# ==============================================================================
echo ""
echo ">>> [RUN 3/4] Executing scripts/eval_dca.py..."
python -u scripts/eval_dca.py 2>&1 | tee results/reviewer_experiments/run3_dca.log
RUN3_STATUS=$?

if [ $RUN3_STATUS -ne 0 ]; then
    echo "ERROR: Run 3 (Decision Curve Analysis) failed with exit code $RUN3_STATUS"
fi

# ==============================================================================
# RUN 4: 10-Decile Calibration Reliability & Recalibration Curves
# ==============================================================================
echo ""
echo ">>> [RUN 4/4] Executing scripts/eval_calibration.py..."
python -u scripts/eval_calibration.py 2>&1 | tee results/reviewer_experiments/run4_calibration.log
RUN4_STATUS=$?

if [ $RUN4_STATUS -ne 0 ]; then
    echo "ERROR: Run 4 (Calibration Diagnostics) failed with exit code $RUN4_STATUS"
fi

conda deactivate 2>/dev/null || true

# Check overall status
OVERALL_STATUS=$(( RUN1_STATUS + RUN2_STATUS + RUN3_STATUS + RUN4_STATUS ))

echo ""
echo "================================================================="
if [ $OVERALL_STATUS -eq 0 ]; then
    echo "  ALL 4 REVIEWER EXPERIMENTS COMPLETED SUCCESSFULLY AT $(date)"
    echo "  Artifacts Generated:"
    echo "   - results/reviewer_experiments/eval_transport_results.csv"
    echo "   - results/reviewer_experiments/eval_capacity_results.csv"
    echo "   - results/reviewer_experiments/eval_dca_results.csv"
    echo "   - results/reviewer_experiments/eval_calibration_deciles.csv"
    echo "   - figures/fig_transport_calibration_drift.pdf & .png"
    echo "   - figures/fig2_dca_net_benefit.pdf & .png"
    echo "   - figures/fig3_calibration_reliability.pdf & .png"
else
    echo "  PIPELINE FINISHED WITH FAILURES (Total Error Status: $OVERALL_STATUS) AT $(date)"
fi
echo "================================================================="

exit $OVERALL_STATUS
