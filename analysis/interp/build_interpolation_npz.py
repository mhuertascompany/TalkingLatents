"""Precompute interpolated FM latents offline and save to NPZ/JSON."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.dataset_comparative import create_comparative_dataloaders, collate_comparative_fn
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build interpolated FM latents")
    parser.add_argument('--json_file', type=str, required=True,
                        help='Path to single-star questions JSON (used for tokenizer only)')
    parser.add_argument('--comparative_json_file', type=str, default=None,
                        help='Path to comparative questions JSON (defaults to --json_file)')
    parser.add_argument('--features_file', type=str, required=True,
                        help='Original FM features .npy used during training')
    parser.add_argument('--split_cache_root', type=str, required=True,
                        help='Root directory with cached splits')
    parser.add_argument('--output_npz', type=str, required=True,
                        help='Output NPZ with interpolated latents')
    parser.add_argument('--output_metadata', type=str, required=True,
                        help='Output JSON with metadata for each interpolation')
    parser.add_argument('--llm_path', type=str, default=None,
                        help='Optional explicit path to LLaMA weights (for tokenizer resolution)')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of comparative samples to process')
    parser.add_argument('--alphas_per_sample', type=int, default=3,
                        help='Number of random α values sampled per dataset example')
    parser.add_argument('--max_total', type=int, default=None,
                        help='Optional cap on total number of interpolations')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--num_spectral_features', type=int, default=8)
    return parser.parse_args()


def make_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def load_dataloader(args: argparse.Namespace, tokenizer_path: str, features: np.ndarray):
    comp_json = args.comparative_json_file or args.json_file
    transforms = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    train_dl, val_dl, test_dl = create_comparative_dataloaders(
        json_file=comp_json,
        features_array=features,
        batch_size=1,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        num_workers=args.num_workers,
        cache_dir=str(Path(args.split_cache_root) / 'comparative' / Path(comp_json).stem),
        tokenizer_path=tokenizer_path,
        max_length=args.max_seq_length,
        num_stellar_features=args.num_spectral_features,
        allow_new_splits=False,
    )
    split_map = {'train': train_dl, 'val': val_dl, 'test': test_dl}
    return split_map[args.split]


def main():
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    features = np.load(args.features_file)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features array, got shape {features.shape}")

    from src.simple_questions import get_model_path
    _, tokenizer_path = get_model_path(args)
    loader = load_dataloader(args, tokenizer_path, features)

    selected_indices = random.sample(range(len(loader.dataset)),
                                     min(args.num_samples, len(loader.dataset)))

    latent_a_list = []
    latent_b_list = []
    dataset_indices = []
    alpha_list = []
    metadata_records = []

    for dataset_idx in selected_indices:
        sample = loader.dataset[dataset_idx]
        batch = collate_comparative_fn([sample])

        base_a = batch['features_a'][0].numpy()
        base_b = batch['features_b'][0].numpy()

        obsid_info = batch.get('obsid', [None])[0]
        if isinstance(obsid_info, (list, tuple)):
            obsid_a, obsid_b = obsid_info
        else:
            obsid_a, obsid_b = obsid_info, None

        star_a_params = make_jsonable(batch.get('star_a_params', [{}])[0])
        star_b_params = make_jsonable(batch.get('star_b_params', [{}])[0])

        alpha_values = np.random.uniform(-1.0, 1.0, size=args.alphas_per_sample)
        if args.max_total is not None:
            remaining = args.max_total - len(latent_a_list)
            if remaining <= 0:
                break
            alpha_values = alpha_values[:remaining]

        for alpha in alpha_values:
            interp_a = base_a + alpha * base_b
            latent_a_list.append(interp_a.astype(np.float32))
            latent_b_list.append(base_b.astype(np.float32))
            dataset_indices.append(dataset_idx)
            alpha_list.append(alpha)
            metadata_records.append({
                'dataset_index': dataset_idx,
                'alpha': alpha,
                'obsid_a': obsid_a,
                'obsid_b': obsid_b,
                'star_a_params': star_a_params,
                'star_b_params': star_b_params,
            })
        if args.max_total is not None and len(latent_a_list) >= args.max_total:
            break

    latent_a = np.stack(latent_a_list)
    latent_b = np.stack(latent_b_list)
    dataset_indices = np.array(dataset_indices, dtype=np.int32)
    alpha_arr = np.array(alpha_list, dtype=np.float32)

    out_npz = Path(args.output_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, latent_a=latent_a, latent_b=latent_b,
             dataset_indices=dataset_indices, alphas=alpha_arr)

    out_meta = Path(args.output_metadata)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(out_meta, 'w') as fh:
        json.dump({'records': metadata_records}, fh, indent=2)

    print(f"Saved interpolations to {out_npz}")
    print(f"Metadata written to {out_meta}")


if __name__ == '__main__':
    main()
