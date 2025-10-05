#!/bin/bash
#SBATCH --job-name=TalkingLatents
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:2
#SBATCH --qos=basic
#SBATCH --mem=64G
#SBATCH --hint=nomultithread


export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:64"}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Paths
JSON_FILE=/data/TalkingLatents/dataset/data/stellar_descriptions_questions_short.json
FEATURES_FILE=/data/TalkingLatents/logs/2025-07-29/features.npy

OUTDIR=/data/TalkingLatents/logs/train_ddp
EXP=llm_mp_loRA_tuned
EPOCHS=10
BATCH=2
MAXLEN=128
WORKERS=8
LR=1e-4
WD=1e-3

srun --kill-on-bad-exit=1 \
      --container-image="/rg/perets_prj/ilay.kamai/containers/lightPred3_C.sqsh" \
     --mem=64G \
     --output=/home/ilay.kamai/athena/logs/TalkingLatents_%j_%s_%t.out \
     --error=/home/ilay.kamai/athena/logs/TalkingLatents_%j_%s_%t.err \
     --no-container-entrypoint \
     --container-mounts=/home/ilay.kamai/work:/data \
    python -u /data/TalkingLatents/src/simple_questions_tuned.py \
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

