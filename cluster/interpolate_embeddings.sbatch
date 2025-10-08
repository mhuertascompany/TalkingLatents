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
LLM_PATH=${LLM_PATH:-/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Meta-Llama-3.1-8B/original}
OUTPUT_DIR=${OUTPUT_DIR:-logs/interp}
EXP=${EXP:-interp_run}
SPLIT=${SPLIT:-val}
BUILD_NUM_SAMPLES=${BUILD_NUM_SAMPLES:-100}
ALPHAS_PER_SAMPLE=${ALPHAS_PER_SAMPLE:-5}
ALPHAS=${ALPHAS:-}
MAX_TOTAL=${MAX_TOTAL:-0}
EVAL_NUM_SAMPLES=${EVAL_NUM_SAMPLES:-25}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-120}
TEMPERATURE=${TEMPERATURE:-0.0}
TOPP=${TOPP:-0.0}
NPZ_PATH=${NPZ_PATH:-${OUTPUT_DIR}/${EXP}_${SPLIT}.npz}
META_PATH=${META_PATH:-${OUTPUT_DIR}/${EXP}_${SPLIT}.metadata.json}
OUTFILE=${OUTPUT_DIR}/${EXP}_${SPLIT}.jsonl
BATCH_SIZE=${BATCH_SIZE:-1}

if [[ -z "${RESUME}" || ! -f "${RESUME}" ]]; then
  echo "ERROR: RESUME checkpoint not found. Set RESUME to a valid *_resume_last.pth path." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
REBUILD=${REBUILD:-1}
if [[ "${REBUILD}" == "1" || ! -f "${NPZ_PATH}" ]]; then
  EXTRA_ALPHA=()
  if [[ -n "${ALPHAS}" ]]; then
    EXTRA_ALPHA=(--alphas "${ALPHAS}")
  fi
  LIMIT_ARG=()
  if [[ "${MAX_TOTAL}" != "0" ]]; then
    LIMIT_ARG=(--max_total "${MAX_TOTAL}")
  fi

  srun --kill-on-bad-exit=1 \
    python -u analysis/interp/build_interpolation_npz.py \
      --json_file "${JSON_FILE}" \
      --comparative_json_file "${COMP_JSON_FILE}" \
      --features_file "${FEATURES_FILE}" \
      --split_cache_root "${SPLIT_CACHE_ROOT}" \
      --output_npz "${NPZ_PATH}" \
      --output_metadata "${META_PATH}" \
      --llm_path "${LLM_PATH}" \
      --split "${SPLIT}" \
      --num_samples "${BUILD_NUM_SAMPLES}" \
      --alphas_per_sample "${ALPHAS_PER_SAMPLE}" \
      --random_seed 42 \
      "${LIMIT_ARG[@]}" \
      "${EXTRA_ALPHA[@]}"
fi

srun --kill-on-bad-exit=1 \
  python -u analysis/interp/eval_interpolation_npz.py \
    --json_file "${JSON_FILE}" \
    --comparative_json_file "${COMP_JSON_FILE}" \
    --split_cache_root "${SPLIT_CACHE_ROOT}" \
    --resume_path "${RESUME}" \
    --interpolation_npz "${NPZ_PATH}" \
    --metadata_json "${META_PATH}" \
    --output "${OUTFILE}" \
    --llm_path "${LLM_PATH}" \
    --split "${SPLIT}" \
    --num_samples "${EVAL_NUM_SAMPLES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOPP}" \
    --num_spectral_features 8 \
    --max_seq_length 128 \
    --batch_size "${BATCH_SIZE}" \
    --random_seed 42

echo "Interpolation outputs written to ${OUTFILE}"
