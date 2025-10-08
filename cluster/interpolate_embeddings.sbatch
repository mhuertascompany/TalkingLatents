#!/bin/bash
# Generate interpolated FM latent evaluations for the two-star LLM

#SBATCH --job-name=tl-interp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH -C v100-32g
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

mkdir -p logs/slurm

module purge
module load pytorch-gpu/py3/2.2.0

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:64"}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Required inputs (override with --export when calling sbatch)
JSON_FILE=${JSON_FILE:-/lustre/fswork/projects/rech/oxl/utl47bv/data/stellar_descriptions_questions_short.json}
COMP_JSON_FILE=${COMP_JSON_FILE:-/lustre/fswork/projects/rech/oxl/utl47bv/data/comparative_dataset.json}
FEATURES_FILE=${FEATURES_FILE:-/lustre/fswork/projects/rech/oxl/utl47bv/data/features.npy}
RESUME=${RESUME:-$(ls -t logs/train_multitok_ddp/*/llm_multitok_resume_last.pth 2>/dev/null | head -1 || true)}
SPLIT_CACHE_ROOT=${SPLIT_CACHE_ROOT:-logs/split_cache}
OUTPUT_DIR=${OUTPUT_DIR:-logs/interp}
EXP=${EXP:-interp_run}
SPLIT=${SPLIT:-val}
NUM_SAMPLES=${NUM_SAMPLES:-25}
ALPHAS=${ALPHAS:--1.0,-0.5,0.0,0.5,1.0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-120}
TEMPERATURE=${TEMPERATURE:-0.0}
TOPP=${TOPP:-0.0}

if [[ -z "${RESUME}" || ! -f "${RESUME}" ]]; then
  echo "ERROR: RESUME checkpoint not found. Set RESUME to a valid *_resume_last.pth path." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
OUTFILE=${OUTPUT_DIR}/${EXP}_${SPLIT}.jsonl

if [[ -z "${ALPHAS}" ]]; then
  echo "ERROR: ALPHAS must be set (e.g. -1.0,-0.5,0,0.5,1.0)" >&2
  exit 1
fi
ALPHA_ARG="--alphas=${ALPHAS}"

srun --kill-on-bad-exit=1 \
  python -u analysis/interp/interpolate_and_generate.py \
    --json_file "${JSON_FILE}" \
    --comparative_json_file "${COMP_JSON_FILE}" \
    --features_file "${FEATURES_FILE}" \
    --split_cache_root "${SPLIT_CACHE_ROOT}" \
    --resume_path "${RESUME}" \
    --output "${OUTFILE}" \
    --split "${SPLIT}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOPP}" \
    --num_spectral_features 8 \
    --max_seq_length 128 \
    --random_seed 42 \
    "${ALPHA_ARG}"

echo "Interpolation outputs written to ${OUTFILE}"
