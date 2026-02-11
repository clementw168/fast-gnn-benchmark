#!/bin/bash
#SBATCH --job-name=train_parts_limit
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=H100 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"


# Execute the Python script with specific arguments
uv run scripts/train_parts_impact.py


# Print job completion time
echo "Job finished at: $(date)"
