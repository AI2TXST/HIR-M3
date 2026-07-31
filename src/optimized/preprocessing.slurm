#!/bin/bash
#SBATCH --job-name=preprocessing_encoding
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=32Gb
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m_e172@txstate.edu
#SBATCH --output=job_pre_output.txt
#SBATCH --error=job_pre_error.txt

echo ""
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

# path to the correct directory
cd "/home/m_e172/OASIS/OASIS_v2/data processing - newest"
echo "Current working directory is `pwd`"

# Load Conda (adjust module name if your cluster uses something different)
module load anaconda

# Activate pre-built Spark environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate spark_env

echo "Starting TX and NB preprocessing in parallel..."
python preprocessing_tx_condensed.py > tx_preprocessing.log 2>&1 &
python preprocessing_nb_condensed.py > nb_preprocessing.log 2>&1 &

wait
echo "Both preprocessing scripts completed."

conda deactivate
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
