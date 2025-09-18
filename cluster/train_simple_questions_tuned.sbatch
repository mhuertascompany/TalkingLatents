#!/bin/bash
# Tuned training on 2x V100 32GB with DDP

#SBATCH --job-name=tl-llm-ddp-tuned
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH -C v100-32g
#SBATCH --partition=gpu_p2
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

module purge
module load pytorch-gpu/py3/2.2.0

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:64"}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Paths
JSON_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/stellar_descriptions_questions_short.json
FEATURES_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/features.npy

OUTDIR=logs/train_ddp
EXP=llm_mp_loRA_tuned
EPOCHS=10
BATCH=2
MAXLEN=128
WORKERS=8
LR=1e-4
WD=1e-3

srun --kill-on-bad-exit=1 \
  python -u src/simple_questions_tuned.py \
    --json_file "${JSON_FILE}" \
    --features_file "${FEATURES_FILE}" \
    --output_dir "${OUTDIR}" \
    --exp_name "${EXP}" \
    --batch_size "${BATCH}" \
    --num_epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --weight_decay "${WD}" \
    --max_seq_length "${MAXLEN}" \
    --num_workers "${WORKERS}" \
    --use_amp \
    --gradient_checkpointing \
    --lora_lr_scale 0.5 \
    --label_smoothing 0.1

