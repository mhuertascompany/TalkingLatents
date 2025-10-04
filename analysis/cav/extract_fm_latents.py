import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.dataset_interpert import StellarQuestionsDataset, collate_fn
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor
from src.simple_questions import _load_spectra_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract FM latent embeddings for concept-labelled samples.")
    parser.add_argument("--json-file", required=True, help="Path to stellar questions JSON file")
    parser.add_argument("--concept-file", required=True, help="Concept bundle produced by prepare_concepts.py")
    parser.add_argument("--split-cache-root", required=True, help="Root directory containing cached split indices")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output-npz", required=True, help="Path to save embeddings/labels npz")
    parser.add_argument("--output-metadata", required=True, help="Path to save detailed metadata JSON")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--num-spectral-features", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--concept-names", nargs="*", default=["main_sequence", "giant"],
                        help="List of concepts to include from the concept file")
    return parser.parse_args()


def resolve_split_cache(args: argparse.Namespace, json_path: Path) -> Path:
    dataset_key = json_path.stem
    split_dir = Path(args.split_cache_root) / "single" / dataset_key
    if not split_dir.exists():
        raise FileNotFoundError(f"Split cache directory not found: {split_dir}")
    return split_dir


def build_dataset(args: argparse.Namespace, json_path: Path, split_dir: Path) -> StellarQuestionsDataset:
    transforms = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    dataset = StellarQuestionsDataset(
        json_file=str(json_path),
        features_array=None,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        spectral_transforms=transforms,
        cache_dir=str(split_dir),
        allow_new_splits=False,
        tokenizer_path=None,
        max_length=args.max_seq_length,
        num_spectral_features=args.num_spectral_features,
    )
    return dataset


def load_concepts(concept_path: Path, concept_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    with open(concept_path, "r") as fh:
        bundle = json.load(fh)
    samples = bundle.get("samples", {})
    selected = {}
    for name in concept_names:
        if name in samples:
            selected[name] = samples[name]
        else:
            print(f"Warning: concept '{name}' not found in concept file")
    if not selected:
        raise ValueError("No matching concepts found in concept file")
    return selected


def collect_indices(concept_samples: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[int], List[int]]:
    indices: List[int] = []
    labels: List[int] = []
    for label_id, (concept, entries) in enumerate(concept_samples.items()):
        for entry in entries:
            indices.append(int(entry["index"]))
            labels.append(label_id)
    return indices, labels


def extract_embeddings(dataset: StellarQuestionsDataset,
                       indices: List[int],
                       labels: List[int],
                       device: torch.device,
                       batch_size: int,
                       concept_names: List[str]) -> Dict[str, Any]:
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    fm = _load_spectra_model().to(device)
    fm.eval()

    all_embeddings: List[np.ndarray] = []
    all_labels: List[int] = []
    all_obsids: List[Any] = []
    extra_meta: List[Dict[str, Any]] = []

    with torch.no_grad():
        cursor = 0
        for batch in loader:
            masked = batch['masked_spectra'].to(device)
            _, _, latent = fm(masked)
            if latent.dim() == 3:
                latent = latent.mean(dim=1)
            latent = latent.detach().cpu().numpy()

            # gather metadata
            obsids = batch.get('obsids', [None] * len(latent))
            stellar_data = batch.get('stellar_data', [{}] * len(latent))

            all_embeddings.append(latent)
            slice_labels = labels[cursor: cursor + len(latent)]
            all_labels.extend(slice_labels)
            all_obsids.extend(obsids)
            for obsid, meta in zip(obsids, stellar_data):
                extra_meta.append({
                    'obsid': obsid,
                    'stellar_data': meta,
                })
            cursor += len(latent)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels_arr = np.array(all_labels, dtype=np.int64)
    obsid_arr = np.array(all_obsids)

    return {
        'embeddings': embeddings,
        'labels': labels_arr,
        'obsids': obsid_arr,
        'metadata': extra_meta,
        'concept_names': concept_names,
    }


def main():
    args = parse_args()
    json_path = Path(args.json_file)
    concept_path = Path(args.concept_file)

    split_dir = resolve_split_cache(args, json_path)
    dataset = build_dataset(args, json_path, split_dir)

    concept_samples = load_concepts(concept_path, args.concept_names)
    indices, labels = collect_indices(concept_samples)

    order = np.argsort(indices)
    indices = [indices[i] for i in order]
    labels = [labels[i] for i in order]

    device = torch.device(args.device)
    result = extract_embeddings(dataset, indices, labels, device, args.batch_size, args.concept_names)

    os.makedirs(Path(args.output_npz).parent, exist_ok=True)
    np.savez(args.output_npz,
             embeddings=result['embeddings'],
             labels=result['labels'],
             obsids=result['obsids'],
             concept_names=np.array(result['concept_names']))

    metadata_records = []
    for idx, meta, label, obsid in zip(indices, result['metadata'], result['labels'], result['obsids']):
        metadata_records.append({
            'dataset_index': idx,
            'concept_id': int(label),
            'concept_name': result['concept_names'][label],
            'obsid': obsid,
            'stellar_data': meta['stellar_data'],
        })

    with open(args.output_metadata, 'w') as fh:
        json.dump({
            'split': args.split,
            'concept_names': result['concept_names'],
            'records': metadata_records,
        }, fh, indent=2)

    print(f"Saved embeddings to {args.output_npz}")
    print(f"Saved metadata to {args.output_metadata}")


if __name__ == '__main__':
    main()
