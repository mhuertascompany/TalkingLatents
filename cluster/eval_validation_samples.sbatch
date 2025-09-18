#!/bin/bash
# Evaluate a trained model on a few validation samples (single GPU)

#SBATCH --job-name=tl-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
# Ensure 32 GB V100s (Jean Zay)
#SBATCH -C v100-32g
#SBATCH --partition=gpu_p2
# Logs
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

# Environment
module purge
module load pytorch-gpu/py3/2.2.0

# Allocator setting compatible with this build
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:64"}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Data/model paths (edit to your filesystem)
JSON_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/stellar_descriptions_questions_short.json
FEATURES_FILE=/lustre/fswork/projects/rech/oxl/utl47bv/data/features.npy
LLM_PATH=/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Meta-Llama-3.1-8B/original

# Output + eval settings
OUTDIR=logs/eval
EXP=eval_run
NUM_SAMPLES=${NUM_SAMPLES:-5}      # override at submit time with --export=NUM_SAMPLES=5
LLM_PREC=${LLM_PREC:-fp16}
MAXLEN=${MAXLEN:-128}
BATCH=${BATCH:-2}
WORKERS=${WORKERS:-0}

mkdir -p "${OUTDIR}"

# You can provide one of these via --export at submit time:
RESUME=/lustre/fswork/projects/rech/oxl/utl47bv/python/TalkingLatents/logs/train_ddp/2025-09-18-16-29/llm_mp_loRA_resume_best.pth
#   WEIGHTS=/path/to/llm_mp_loRA.pth            (plain weights)

if [[ -n "${RESUME:-}" ]]; then
  echo "Evaluating with composite checkpoint: ${RESUME}"
  srun --kill-on-bad-exit=1 \
    python -u src/eval_validation_samples.py \
      --llm_path "${LLM_PATH}" \
      --json_file "${JSON_FILE}" \
      --features_file "${FEATURES_FILE}" \
      --output_dir "${OUTDIR}" \
      --exp_name "${EXP}" \
      --resume_path "${RESUME}" \
      --num_samples "${NUM_SAMPLES}" \
      --llm_precision "${LLM_PREC}" \
      --batch_size "${BATCH}" \
      --max_seq_length "${MAXLEN}" \
      --num_workers "${WORKERS}"
elif [[ -n "${WEIGHTS:-}" ]]; then
  echo "Evaluating with plain weights: ${WEIGHTS}"
  srun --kill-on-bad-exit=1 \
    python -u src/eval_validation_samples.py \
      --llm_path "${LLM_PATH}" \
      --json_file "${JSON_FILE}" \
      --features_file "${FEATURES_FILE}" \
      --output_dir "${OUTDIR}" \
      --exp_name "${EXP}" \
      --weights "${WEIGHTS}" \
      --num_samples "${NUM_SAMPLES}" \
      --llm_precision "${LLM_PREC}" \
      --batch_size "${BATCH}" \
      --max_seq_length "${MAXLEN}" \
      --num_workers "${WORKERS}"
else
  echo "Evaluating with LLaMA base weights from: ${LLM_PATH}"
  srun --kill-on-bad-exit=1 \
    python -u src/eval_validation_samples.py \
      --llm_path "${LLM_PATH}" \
      --json_file "${JSON_FILE}" \
      --features_file "${FEATURES_FILE}" \
      --output_dir "${OUTDIR}" \
      --exp_name "${EXP}" \
      --num_samples "${NUM_SAMPLES}" \
      --llm_precision "${LLM_PREC}" \
      --batch_size "${BATCH}" \
      --max_seq_length "${MAXLEN}" \
      --num_workers "${WORKERS}"
fi

