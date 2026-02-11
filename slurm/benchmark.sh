#!/bin/bash
#SBATCH --job-name=sampling_benchmark
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=24:00:00

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"


# Execute the Python script with specific arguments
uv run scripts/sampling_benchmark.py

# Print job completion time
echo "Job finished at: $(date)"
