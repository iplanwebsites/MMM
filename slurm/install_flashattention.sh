#!/bin/bash

# FlashAttention needs CUDA to be installed, hence the installation must be run on compute nodes.
# Nodes of Compute Canada cannot access internet, so FlashAttention must be installed from source.

# Set SLURM / hardware environment
#SBATCH --job-name=install-fa
#SBATCH --output=logs/install-fa.out
#SBATCH --error=logs/install-fa_err.out
#SBATCH --account=def-pasquier
#SBATCH --mail-user=raa60@sfu.ca # Default mail
#SBATCH --nodes=1            # total nb of nodes
#SBATCH --ntasks-per-node=1  # nb of tasks per node
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6   # nb of CPU cores per task
#SBATCH --time=2:00:00

# Output GPUs and ram info
echo "START TIME: $(date)"
nvidia-smi
nvidia-smi topo -m
free -m

# Load the virtual environment and install FlashAttention from source
source .venv/bin/activate
cd $HOME/flash-attention
pip install .

echo "END TIME: $(date)"
