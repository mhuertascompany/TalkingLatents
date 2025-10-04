import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare concept-specific sample lists from the stellar questions dataset."
    )
    parser.add_argument("--json-file", required=True, help="Path to stellar questions JSON file")
    parser.add_argument("--split-cache-root", required=True, help="Root directory with cached split indices")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train",
                        help="Which split to export")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed used when the splits were generated")
    parser.add_argument("--output", required=True, help="Path to save the concept JSON output")
    parser.add_argument("--main-sequence-min-logg", type=float, default=3.5,
                        help="Lower bound on logg for main-sequence classification")
    parser.add_argument("--giant-max-logg", type=float, default=3.2,
                        help="Upper bound on logg for giant classification")
    parser.add_argument("--min-per-concept", type=int, default=50,
                        help="Warn if fewer samples than this threshold are collected per concept")
    return parser.parse_args()


def load_dataset(json_path: Path) -> List[Dict[str, Any]]:
    with open(json_path, "r") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        elif "questions" in data:
            data = data["questions"]
        else:
            raise ValueError("Unexpected JSON structure: missing 'data' or 'questions' key")
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of samples")
    return data


def locate_split_indices(args: argparse.Namespace, json_path: Path, num_samples: int) -> Dict[str, np.ndarray]:
    dataset_key = json_path.stem
    split_root = Path(args.split_cache_root) / "single" / dataset_key
    cache_key = f"{dataset_key}_{num_samples}_{args.train_ratio}_{args.val_ratio}_{args.test_ratio}_{args.random_seed}"
    split_file = split_root / f"splits_{cache_key}.npz"
    if not split_file.exists():
        available = list(split_root.glob("splits_*.npz"))
        msg = [
            f"Expected split cache not found: {split_file}",
            "Available cache files:" if available else "No cache files found in split directory.",
        ]
    # ... the script continues ...
        if available:
            msg.extend([f"  - {path}" for path in available])
        raise FileNotFoundError("\n".join(msg))

    cached = np.load(split_file, allow_pickle=True)
    return {key: cached[key] for key in ("train_indices", "val_indices", "test_indices")}


def classify_sample(sample: Dict[str, Any], ms_min_logg: float, giant_max_logg: float) -> str:
    stellar = sample.get("stellar_data", {}) or {}
    logg = None
    for key in ("logg", "log_g", "log_g_spectroscopic", "logg_spectroscopic"):
        if key in stellar:
            logg = stellar[key]
            break
    if logg is None:
        raise ValueError("Sample missing logg in stellar_data")

    try:
        logg_val = float(logg)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to interpret logg value '{logg}'") from exc

    if logg_val >= ms_min_logg:
        return "main_sequence"
    if logg_val <= giant_max_logg:
        return "giant"
    return "ambiguous"


def build_concepts(args: argparse.Namespace) -> Dict[str, Any]:
    json_path = Path(args.json_file)
    samples = load_dataset(json_path)
    indices = locate_split_indices(args, json_path, len(samples))

    split_map = {
        "train": indices["train_indices"],
        "val": indices["val_indices"],
        "test": indices["test_indices"],
    }
    target_indices = split_map[args.split]

    concepts = {"main_sequence": [], "giant": [], "ambiguous": []}

    for idx in target_indices:
        sample = samples[int(idx)]
        try:
            label = classify_sample(sample, args.main_sequence_min_logg, args.giant_max_logg)
        except ValueError as err:
            print(f"Skipping sample {sample.get('obsid')} due to missing metadata: {err}")
            continue

        entry = {
            "obsid": sample.get("obsid"),
            "logg": sample.get("stellar_data", {}).get("logg"),
            "stellar_data": sample.get("stellar_data", {}),
            "question": sample.get("description", ""),
            "index": int(idx),
        }
        concepts[label].append(entry)

    for concept in ("main_sequence", "giant"):
        count = len(concepts[concept])
        if count < args.min_per_concept:
            print(f"Warning: only {count} samples collected for concept '{concept}'")

    return {
        "concept": "main_sequence_vs_giant",
        "split": args.split,
        "thresholds": {
            "main_sequence_min_logg": args.main_sequence_min_logg,
            "giant_max_logg": args.giant_max_logg,
        },
        "counts": {k: len(v) for k, v in concepts.items()},
        "samples": concepts,
    }


def main():
    args = parse_args()
    os.makedirs(Path(args.output).parent, exist_ok=True)
    concept_bundle = build_concepts(args)
    with open(args.output, "w") as fh:
        json.dump(concept_bundle, fh, indent=2)
    print(f"Saved concept bundle to {args.output}")
    print(json.dumps(concept_bundle["counts"], indent=2))


if __name__ == "__main__":
    main()
