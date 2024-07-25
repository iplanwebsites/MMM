#!/bin/bash

# Inspired from https://github.com/bigscience-workshop/bigscience/blob/7ccf7e42577fe71e88cf8bed3b9ca965c7afb8f7/train/tr11-176B-ml/tr11-176B-ml.slurm

# Set SLURM / hardware environment
#SBATCH --job-name=train-mmm
#SBATCH --output=logs/train-mmm.out
#SBATCH --error=logs/train-mmm_err.out
#SBATCH --account=def-pasquier
#SBATCH --mail-user=raa60@sfu.ca # Default mail
#SBATCH --nodes=1            # total nb of nodes
#SBATCH --ntasks-per-node=1  # nb of tasks per node
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=48   # nb of CPU cores per task
#SBATCH --mem=40G
#SBATCH --time=24:00:00

# Define args
MODEL_TRAIN_ARGS=" \
    --deepspeed \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 16 \
    "

# Output GPUs and ram info
echo "START TIME: $(date)"
nvidia-smi
nvidia-smi topo -m
free -h

# Hardware vars
GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=9902
echo "Master addr: $MASTER_ADDR"
echo "Node list: $SLURM_JOB_NODELIST"

# Defining the right environment variables
export PYTHONPATH=$HOME/MMM
export HF_HOME=$SLURM_TMPDIR/.hf_cache
export HF_METRICS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
# The below variable is required to avoid a warning with the hf tokenizers lib and multiprocessing
# Weirdly, the tokenizer lib is used somewhere before that the dataloader create several workers,
# even when average_num_tokens_per_note is hardcoded in the Dataset class
# https://github.com/huggingface/transformers/issues/5486
# best explanation: https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
export TOKENIZERS_PARALLELISM=0

# Move hugging face dataset from scratch to local file system
# This is done on every nodes.
# Docs: https://docs.alliancecan.ca/wiki/Using_node-local_storage
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 cp -R $SCRATCH/data/GigaMIDI $SLURM_TMPDIR/data/

# Set launcher command with params
export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role $SLURMD_NODENAME: \
    --tee 3 \
    "

# Load the python environment
source .venv/bin/activate

# Run the training
srun --jobid "$SLURM_JOBID" bash -c "$LAUNCHER scripts/train_model.py $MODEL_TRAIN_ARGS"

echo "END TIME: $(date)"
