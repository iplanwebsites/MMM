#!/bin/bash

# Set SLURM / hardware environment
#SBATCH --job-name=preprocess-data
#SBATCH --output=logs/preprocess-data.out
#SBATCH --error=logs/preprocess-data_err.out
#SBATCH --account=def-pasquier
#SBATCH --mail-user=raa60@sfu.ca # Default mail
#SBATCH --nodes=1            # total nb of nodes
#SBATCH --ntasks-per-node=1  # nb of tasks per node
#SBATCH --cpus-per-task=64    # nb of CPU cores per task
#SBATCH --time=4:00:00

# Output ram info
echo "START TIME: $(date)"
free -m

# Defining the right environment variables
export PYTHONPATH=$WORK/MMM
export HF_HOME=$SCRATCH/.hf_cache

# Load the python environment
# Make sure the required packages are installed
source .venv/bin/activate

# Run the training
srun --jobid "$SLURM_JOBID" bash -c "python3.9 clean_mmd.py"
srun --jobid "$SLURM_JOBID" bash -c "python3.9 tokenize_datasets.py"
srun --jobid "$SLURM_JOBID" bash -c "python3.9 gather_tokens_paths_per_genre.py"

echo "END TIME: $(date)"
