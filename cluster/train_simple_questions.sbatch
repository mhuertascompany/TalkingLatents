#!/bin/bash
# Real training on 2x V100 32GB with DDP

#SBATCH --job-name=tl-llm-ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3           # 2 ranks = 2 GPUs
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=30            # tune for your dataloader/tokenization
#SBATCH --hint=nomultithread
#SBATCH --time=23:00:00               # adjust as needed

# Ensure 32 GB V100s (Jean Zay)
#SBATCH -C v100-32g
# Logs
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# If you hit host RAM OOM during checkpoint load, consider uncommenting:
#SBATCH --mem=128G

set -euo pipefail

mkdir -p logs/slurm

# Load environment (adjust to your site)
module purge
module load pytorch-gpu/py3/2.2.0



# NCCL / allocator tuning
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True,max_split_size_mb=64"}

# Data/model paths (edit to your filesystem)
JSON_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/stellar_descriptions_questions_short.json
FEATURES_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/features.npy
LLM_PATH=/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Meta-Llama-3.1-8B/original

# Training knobs
OUTDIR=logs/train_ddp
EXP=llm_mp_loRA
EPOCHS=10
BATCH=2              # per-GPU
MAXLEN=128
WORKERS=8
LR=1e-4

mkdir -p "${OUTDIR}"

# Launch one Python process per task/GPU
srun --kill-on-bad-exit=1 \
  python -u src/simple_questions.py \
    --llm_path ${LLM_PATH} \
    --json_file ${JSON_FILE} \
    --features_file ${FEATURES_FILE} \
    --output_dir ${OUTDIR} \
    --exp_name ${EXP} \
    --batch_size ${BATCH} \
    --num_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --max_seq_length ${MAXLEN} \
    --num_workers ${WORKERS} \
    --llm_precision fp16 \
    --use_amp

