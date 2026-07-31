#!/bin/bash
#SBATCH --job-name=oasis_prep_condensed
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=32Gb
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_prep_condensed_output.txt
#SBATCH --error=job_prep_condensed_error.txt

echo ""
echo "Starting Condensed PySpark Preprocessing at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

# Change directory to working workspace
cd "/home/m_e172/OASIS/OASIS_v2/data processing - newest"
echo "Current working directory is `pwd`"

# Robust Conda Environment Initialization
source ~/.bashrc 2>/dev/null || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate spark_env 2>/dev/null || source activate spark_env 2>/dev/null || true

# Thread & PySpark memory settings
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "Launching Texas & Nationwide Condensed PySpark Preprocessing Pipelines in parallel..."

# Run Texas sub-cohort and Nationwide cohort condensed preprocessing in parallel
python preprocessing_tx_condensed.py > log_tx_condensed.out 2>&1 &
python preprocessing_nb_condensed.py > log_nb_condensed.out 2>&1 &

# Wait for both background processes to complete
wait

conda deactivate
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
