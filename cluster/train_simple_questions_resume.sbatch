#!/bin/bash
# Resume training on 2x V100 32GB with DDP from the latest composite checkpoint

#SBATCH --job-name=tl-llm-ddp-resume
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2           # 2 ranks = 2 GPUs
#SBATCH --gres=gpu:2                  # request 2 GPUs on the node
#SBATCH --cpus-per-task=20            # tune for your dataloader/tokenization
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00               # adjust as needed
# Ensure 32 GB V100s (Jean Zay)
#SBATCH -C v100-32g
# Logs
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

# Load environment (adjust to your site)
module purge
module load pytorch-gpu/py3/2.2.0

# CUDA allocator: compatible setting for this build
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:64"}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Data/model paths (edit to your filesystem)
JSON_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/stellar_descriptions_questions_short.json
FEATURES_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/features.npy
LLM_PATH=/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Meta-Llama-3.1-8B/original

# Training knobs
OUTDIR=logs/train_ddp
EXP=llm_mp_loRA
EPOCHS=50
BATCH=2              # per-GPU
MAXLEN=128
WORKERS=8
LR=1e-4

# Auto-detect the latest composite checkpoint (resume_last preferred)
RESUME=${RESUME:-$(ls -t ${OUTDIR}/*/${EXP}_resume_last.pth 2>/dev/null | head -1 || true)}
if [[ -z "${RESUME}" ]]; then
  # Fallback to "best" composite checkpoint if available
  RESUME=$(ls -t ${OUTDIR}/*/${EXP}_resume_best.pth 2>/dev/null | head -1 || true)
fi

echo "Detected resume checkpoint: ${RESUME:-<none>}"

# Build common args
COMMON_ARGS=(
  --llm_path "${LLM_PATH}"
  --json_file "${JSON_FILE}"
  --features_file "${FEATURES_FILE}"
  --output_dir "${OUTDIR}"
  --exp_name "${EXP}"
  --batch_size "${BATCH}"
  --num_epochs "${EPOCHS}"
  --learning_rate "${LR}"
  --max_seq_length "${MAXLEN}"
  --num_workers "${WORKERS}"
  --llm_precision fp16
  --use_amp
)

# If we found a composite resume checkpoint, resume exactly; else warm-start from last weights
if [[ -n "${RESUME}" && -f "${RESUME}" ]]; then
  echo "Resuming exactly from ${RESUME}"
  srun --kill-on-bad-exit=1 \
    python -u src/simple_questions.py \
      "${COMMON_ARGS[@]}" \
      --resume_path "${RESUME}"
else
  # Warm-start path: use the latest run dir as checkpoint_dir for weights only; disable warmup
  LAST_RUN_DIR=$(ls -dt ${OUTDIR}/* 2>/dev/null | head -1 || true)
  echo "No composite checkpoint found; warm-starting from weights in: ${LAST_RUN_DIR:-<none>}"
  srun --kill-on-bad-exit=1 \
    python -u src/simple_questions.py \
      "${COMMON_ARGS[@]}" \
      --checkpoint_dir "${LAST_RUN_DIR}" \
      --warmup_epochs 0
fi

