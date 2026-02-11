#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=A100,L40S,A40 # A100,L40S,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"


echo "SWEEP_ID=$SWEEP_ID"

# Execute the Python script with specific arguments
uv run wandb agent clement_wang/gnn_experiments/$SWEEP_ID

# Print job completion time
echo "Job finished at: $(date)"
