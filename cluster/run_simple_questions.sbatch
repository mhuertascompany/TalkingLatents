#!/bin/bash
# Example SLURM batch script for running src/simple_questions.py on V100 GPUs
# Adjust account/partition/paths to your Jean Zay project settings.

#SBATCH --job-name=simple_questions
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 4 tasks = 4 GPUs on a single node
#SBATCH --gres=gpu:v100:4            # Request 4x V100
#SBATCH --cpus-per-task=10           # Tune per your data pipeline
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00              # Adjust time limit
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Optional: specify partition/constraint as required by your site
#SBATCH --partition=gpu              # e.g., gpu for Jean Zay; update if needed
#SBATCH --constraint=v100            # ensure V100; update to site-specific value if needed

set -euo pipefail

# Create logs dir for slurm output if missing
mkdir -p logs

echo "Running on nodes: ${SLURM_JOB_NODELIST}"
echo "GPUs per node: ${SLURM_GPUS_ON_NODE:-unknown}"

# Load your environment (one of the following):

# 1) Module-based PyTorch (example; adapt to your site modules)
# module purge
# module load pytorch-gpu/py3/2.2.0

# 2) Conda environment (recommended)
# source ~/.bashrc
# conda activate talking-latents

# NCCL debug (optional)
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Optional NCCL envs for some clusters (uncomment if needed)
# export NCCL_IB_DISABLE=0
# export NCCL_P2P_DISABLE=0
# export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"^lo"}

# Root of your LLaMA models (override here or pass via --llm_root)
export LLM_ROOT=/data/.llama

# Data/config paths (edit for your filesystem)
JSON_FILE=/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json
FEATURES_FILE=/data/TalkingLatents/logs/2025-07-29/features.npy

# Output directory
OUTDIR=logs

# Run with srun: one Python process per task/GPU
srun --kill-on-bad-exit=1 \
  python -u src/simple_questions.py \
    --json_file ${JSON_FILE} \
    --features_file ${FEATURES_FILE} \
    --output_dir ${OUTDIR} \
    --exp_name interpert \
    --llm_root ${LLM_ROOT} \
    --llm_model Llama3.1-8B \
    --batch_size 4 \
    --num_epochs 1000 \
    --learning_rate 1e-4 \
    --freeze_llm \
    --freeze_spectral

