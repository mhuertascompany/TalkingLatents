"""CAV analysis helper.

This module wraps the three major steps:
  1) prepare concept labels (main-sequence vs. giant) from the cached split
  2) extract FM latent embeddings for those samples
  3) train a linear probe to obtain concept activation vectors (CAVs)
Optionally step 4 invokes the perturb-and-generate script to inspect text sensitivity.

Usage (run on the Jean Zay GPU node where data/checkpoints live):
    python analysis/cav/run_cav_pipeline.py \
        --json-file /lustre/.../stellar_descriptions_questions_short.json \
        --split-cache-root logs/split_cache \
        --resume-path logs/train_multitok_ddp/.../llm_multitok_resume_last.pth \
        --output-root analysis/cav/runs/ms_vs_giant

After completion you will have:
  - concept bundle JSON
  - latent embeddings NPZ + metadata JSON
  - trained CAV NPZ (weights, intercept, metrics)
  - optional perturbation output (JSONL) if --run-perturbation is set

No external imports beyond numpy/sklearn/scipy are required.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

PREP_SCRIPT = ROOT / "analysis" / "cav" / "prepare_concepts.py"
LATENT_SCRIPT = ROOT / "analysis" / "cav" / "extract_fm_latents.py"
PERTURB_SCRIPT = ROOT / "analysis" / "cav" / "perturb_and_generate.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end CAV pipeline for main sequence vs giant analysis")
    parser.add_argument("--json-file", required=True, help="Path to stellar JSON dataset")
    parser.add_argument("--split-cache-root", required=True, help="Cached split directory")
    parser.add_argument("--resume-path", required=True, help="Composite checkpoint for the multimodal model")
    parser.add_argument("--output-root", required=True, help="Base output directory for artifacts")

    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--main-sequence-min-logg", type=float, default=3.5)
    parser.add_argument("--giant-max-logg", type=float, default=3.2)

    parser.add_argument("--num-spectral-features", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test split ratio for logistic regression validation")
    parser.add_argument("--logreg-max-iter", type=int, default=1000)

    parser.add_argument("--run-perturbation", action="store_true",
                        help="If set, run perturbation generation after training CAV")
    parser.add_argument("--alphas", default="-1.5,-0.5,0,0.5,1.5")
    parser.add_argument("--num-perturb-samples", type=int, default=10)
    parser.add_argument("--perturb-split", choices=["train", "val", "test"], default="val")

    return parser.parse_args()


def run_subprocess(cmd: List[str]):
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def step_prepare_concepts(args: argparse.Namespace, out_dir: Path) -> Path:
    concept_path = out_dir / "concepts" / f"ms_vs_giant_{args.split}.json"
    concept_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(PREP_SCRIPT),
        "--json-file", str(args.json_file),
        "--split-cache-root", str(args.split_cache_root),
        "--split", args.split,
        "--output", str(concept_path),
        "--train-ratio", str(args.train_ratio),
        "--val-ratio", str(args.val_ratio),
        "--test-ratio", str(args.test_ratio),
        "--random-seed", str(args.random_seed),
        "--main-sequence-min-logg", str(args.main_sequence_min_logg),
        "--giant-max-logg", str(args.giant_max_logg),
    ]
    run_subprocess(cmd)
    return concept_path


def step_extract_latents(args: argparse.Namespace, concept_path: Path, out_dir: Path) -> Dict[str, Path]:
    latent_dir = out_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    npz_path = latent_dir / f"ms_vs_giant_{args.split}.npz"
    meta_path = latent_dir / f"ms_vs_giant_{args.split}.metadata.json"
    cmd = [
        sys.executable, str(LATENT_SCRIPT),
        "--json-file", str(args.json_file),
        "--concept-file", str(concept_path),
        "--split-cache-root", str(args.split_cache_root),
        "--split", args.split,
        "--output-npz", str(npz_path),
        "--output-metadata", str(meta_path),
        "--train-ratio", str(args.train_ratio),
        "--val-ratio", str(args.val_ratio),
        "--test-ratio", str(args.test_ratio),
        "--random-seed", str(args.random_seed),
        "--num-spectral-features", str(args.num_spectral_features),
        "--max-seq-length", str(args.max_seq_length),
        "--batch-size", str(args.batch_size),
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    run_subprocess(cmd)
    return {"npz": npz_path, "metadata": meta_path}


def step_train_cav(latent_npz: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    data = np.load(latent_npz)
    embeddings = data['embeddings']
    labels = data['labels']
    concept_names = list(data['concept_names'])
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Embeddings and labels length mismatch")

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=args.test_size, random_state=args.random_seed, stratify=labels
    )

    clf = LogisticRegression(max_iter=args.logreg_max_iter)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=concept_names))

    cav_vector = clf.coef_.ravel().astype(np.float32)
    cav_norm = np.linalg.norm(cav_vector)
    if cav_norm == 0:
        raise ValueError("CAV weight norm is zero")

    cav_dir = cav_vector / cav_norm

    cav_path = out_dir / "cavs" / f"ms_vs_giant_{args.split}.npz"
    cav_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cav_path,
             vector=cav_dir,
             raw_weights=cav_vector,
             intercept=clf.intercept_.astype(np.float32),
             accuracy=acc,
             concept_names=np.array(concept_names),
             random_seed=args.random_seed,
             test_size=args.test_size)
    print(f"Saved CAV to {cav_path}")
    return cav_path


def step_perturb_and_generate(args: argparse.Namespace, concept_path: Path,
                              latent_paths: Dict[str, Path], cav_path: Path, out_dir: Path):
    output_dir = out_dir / "perturbation"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = output_dir / f"ms_vs_giant_{args.perturb_split}.jsonl"

    cmd = [
        sys.executable, str(PERTURB_SCRIPT),
        "--json-file", str(args.json_file),
        "--concept-file", str(concept_path),
        "--latents-npz", str(latent_paths['npz']),
        "--metadata-json", str(latent_paths['metadata']),
        "--cav-file", str(cav_path),
        "--resume-path", str(args.resume_path),
        "--split-cache-root", str(args.split_cache_root),
        "--output", str(out_jsonl),
        "--split", args.perturb_split,
        "--alphas", args.alphas,
        "--num-samples", str(args.num_perturb_samples),
        "--num-spectral-features", str(args.num_spectral_features),
        "--max-seq-length", str(args.max_seq_length),
        "--hidden-dim", "512",
        "--temperature", "0.0",
        "--top-p", "0.0",
    ]
    run_subprocess(cmd)


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Step 1: preparing concept labels")
    concept_path = step_prepare_concepts(args, out_root)

    print("Step 2: extracting FM latents")
    latent_paths = step_extract_latents(args, concept_path, out_root)

    print("Step 3: training logistic regression CAV")
    cav_path = step_train_cav(latent_paths['npz'], out_root, args)

    if args.run_perturbation:
        print("Step 4: generating perturbed outputs")
        step_perturb_and_generate(args, concept_path, latent_paths, cav_path, out_root)
    else:
        print("Skipping perturbation step (use --run-perturbation to enable)")

    print("Pipeline complete. Artifacts stored under", out_root)


if __name__ == "__main__":
    main()
