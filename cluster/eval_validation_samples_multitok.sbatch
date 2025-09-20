#!/bin/bash
# Evaluate multi-token model on a random subset (single GPU)

#SBATCH --job-name=tl-eval-multitok
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH -C v100-32g
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
LLM_PATH=/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Meta-Llama-3.1-8B/original

# Eval knobs (override with --export=VAR=VALUE)
OUTDIR=${OUTDIR:-logs/eval_multitok}
EXP=${EXP:-eval_multitok}
NUM_SAMPLES=${NUM_SAMPLES:-5}
TOKS=${TOKS:-4}
MAXLEN=${MAXLEN:-96}
TOKENS=${TOKENS:-100}          # generation length
TEMP=${TEMP:-0.2}
TOPP=${TOPP:-0.8}
SEED=${SEED:-42}
LLM_PREC=${LLM_PREC:-fp16}

mkdir -p "${OUTDIR}"

# You can provide one of:
#   RESUME=/path/to/llm_multitok_resume_best.pth (composite)
#   WEIGHTS=/path/to/llm_multitok.pth           (plain weights)

SR_ARGS=(
  --llm_path "${LLM_PATH}"
  --json_file "${JSON_FILE}"
  --features_file "${FEATURES_FILE}"
  --output_dir "${OUTDIR}"
  --exp_name "${EXP}"
  --num_spectral_features "${TOKS}"
  --max_seq_length "${MAXLEN}"
  --num_samples "${NUM_SAMPLES}"
  --max_new_tokens "${TOKENS}"
  --temperature "${TEMP}"
  --top_p "${TOPP}"
  --random_seed "${SEED}"
  --llm_precision "${LLM_PREC}"
)

if [[ -n "${RESUME:-}" ]]; then
  srun --kill-on-bad-exit=1 \
    python -u src/eval_validation_samples_multitok.py \
      "${SR_ARGS[@]}" \
      --resume_path "${RESUME}"
elif [[ -n "${WEIGHTS:-}" ]]; then
  srun --kill-on-bad-exit=1 \
    python -u src/eval_validation_samples_multitok.py \
      "${SR_ARGS[@]}" \
      --weights "${WEIGHTS}"
else
  srun --kill-on-bad-exit=1 \
    python -u src/eval_validation_samples_multitok.py \
      "${SR_ARGS[@]}"
fi

