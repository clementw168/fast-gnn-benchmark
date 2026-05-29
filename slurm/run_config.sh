#!/bin/bash
#SBATCH --job-name=run_config
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=audible,A100,L40S # audible,A100,L40S,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"


echo "CONFIG_FILE=$CONFIG_FILE"

# Execute the Python script with specific arguments
uv run scripts/main.py -c $CONFIG_FILE

# Print job completion time
echo "Job finished at: $(date)"
