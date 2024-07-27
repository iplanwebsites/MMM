#!/bin/bash

# Script to train a model while sending consecutive Slurm jobs

# Set vars
TEST_RESULTS_FILE="runs/MMM/test_results.json"
COUNT=0

# Loop job until training is done
while [ ! -f "$TEST_RESULTS_FILE" ];
do
    sbatch --wait slurm/train_model.sh
    mv "logs/train-mmm.out" "train-mmm_$COUNT.out"
    mv "logs/train-mmm_err.out" "train-mmm_${COUNT}_err.out"
    ((COUNT++))
done
