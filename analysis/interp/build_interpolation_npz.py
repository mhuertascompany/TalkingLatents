"""Build a reusable NPZ/JSON bundle of interpolated FM latents."""

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
from data.dataset_interpert import create_stellar_dataloaders, collate_fn as single_collate_fn
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor


def make_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build interpolated FM latents")
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--comparative_json_file', type=str, default=None)
    parser.add_argument('--features_file', type=str, required=True,
                        help='Original FM features array used in training')
    parser.add_argument('--split_cache_root', type=str, required=True)
    parser.add_argument('--output_npz', type=str, required=True)
    parser.add_argument('--output_metadata', type=str, required=True)
    parser.add_argument('--llm_path', type=str, default=None,
                        help='Unused but kept for CLI symmetry')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--dataset', choices=['comparative', 'single'], default='comparative',
                        help='Which dataset type to draw samples from when building NPZ')
    parser.add_argument('--build_num_samples', type=int, default=100,
                        help='How many dataset rows to interpolate before random sub-sampling')
    parser.add_argument('--alphas_per_sample', type=int, default=5,
                        help='Number of random α values in [-1,1] per dataset row')
    parser.add_argument('--max_total', type=int, default=None,
                        help='Optional hard cap on total interpolations written to NPZ')
    parser.add_argument('--alphas', type=str, default=None,
                        help='Optional comma/space separated list of alpha values to use instead of random sampling')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--num_spectral_features', type=int, default=8)
    return parser.parse_args()


def load_loader(args: argparse.Namespace, tokenizer_path: str, features: np.ndarray):
    transforms = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])

    if args.dataset == 'single':
        cache_base = Path(args.split_cache_root) / 'single' / Path(args.json_file).stem
        train_dl, val_dl, test_dl = create_stellar_dataloaders(
            json_file=args.json_file,
            features_array=features,
            spectral_transforms=transforms,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_seed,
            num_spectral_features=args.num_spectral_features,
            cache_dir=str(cache_base),
            allow_new_splits=False,
            tokenizer_path=tokenizer_path,
            max_length=args.max_seq_length,
            batch_size=1,
            num_workers=args.num_workers,
        )
    else:
        comp_json = args.comparative_json_file or args.json_file
        cache_base = Path(args.split_cache_root) / 'comparative' / Path(comp_json).stem
        train_dl, val_dl, test_dl = create_comparative_dataloaders(
            json_file=comp_json,
            features_array=features,
            batch_size=1,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_seed,
            num_workers=args.num_workers,
            cache_dir=str(cache_base),
            tokenizer_path=tokenizer_path,
            max_length=args.max_seq_length,
            num_stellar_features=args.num_spectral_features,
            allow_new_splits=False,
        )

    loaders = {'train': train_dl, 'val': val_dl, 'test': test_dl}
    return loaders[args.split]


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
    loader = load_loader(args, tokenizer_path, features)

    selected_indices = random.sample(range(len(loader.dataset)),
                                     min(args.build_num_samples, len(loader.dataset)))

    latent_list = []
    dataset_indices = []
    alpha_list = []
    metadata_records = []

    total_cap = args.max_total if args.max_total is not None else float('inf')

    alpha_values = None
    if args.alphas:
        raw = args.alphas.replace(',', ' ').split()
        if not raw:
            raise ValueError('Provided --alphas is empty after parsing')
        alpha_values = [float(val) for val in raw]
        print(f"Using explicit alpha values: {alpha_values}")

    for dataset_idx in selected_indices:
        if len(latent_list) >= total_cap:
            break
        sample = loader.dataset[dataset_idx]
        if args.dataset == 'single':
            batch = single_collate_fn([sample])
        else:
            batch = collate_comparative_fn([sample])

        def to_index(val):
            if isinstance(val, torch.Tensor):
                return int(val.item())
            if isinstance(val, (list, tuple, np.ndarray)):
                return int(val[0])
            return int(val)

        if args.dataset == 'single':
            latent_a = batch['masked_spectra'][0].cpu().numpy().astype(np.float32)
            partner_idx = random.randrange(len(loader.dataset))
            if len(loader.dataset) > 1:
                while partner_idx == dataset_idx:
                    partner_idx = random.randrange(len(loader.dataset))
            partner_sample = loader.dataset[partner_idx]
            partner_batch = single_collate_fn([partner_sample])
            latent_b = partner_batch['masked_spectra'][0].cpu().numpy().astype(np.float32)

            obsid_a = batch.get('obsids', [None])[0]
            obsid_b = partner_batch.get('obsids', [None])[0]
            star_a_params = make_jsonable(batch.get('stellar_data', [{}])[0])
            star_b_params = make_jsonable(partner_batch.get('stellar_data', [{}])[0])
        else:
            if 'masked_spectra_a' in batch and 'masked_spectra_b' in batch:
                latent_a = batch['masked_spectra_a'][0].cpu().numpy().astype(np.float32)
                latent_b = batch['masked_spectra_b'][0].cpu().numpy().astype(np.float32)
            else:
                if 'features_a_indices' in batch:
                    idx_a = to_index(batch['features_a_indices'][0])
                    idx_b = to_index(batch['features_b_indices'][0])
                else:
                    idx_a = to_index(batch.get('star_a_df_index', [0])[0])
                    idx_b = to_index(batch.get('star_b_df_index', [0])[0])
                latent_a = features[int(idx_a)].astype(np.float32)
                latent_b = features[int(idx_b)].astype(np.float32)

            obsid_info = batch.get('obsid', [None])[0]
            if isinstance(obsid_info, (list, tuple)):
                obsid_a, obsid_b = obsid_info
            else:
                obsid_a, obsid_b = obsid_info, None

            star_a_params = make_jsonable(batch.get('star_a_params', [{}])[0])
            star_b_params = make_jsonable(batch.get('star_b_params', [{}])[0])

        if alpha_values is not None:
            alphas = alpha_values
        else:
            alphas = np.random.uniform(-1.0, 1.0, size=args.alphas_per_sample)
        for alpha in alphas:
            if len(latent_list) >= total_cap:
                break
            interp = latent_a + float(alpha) * latent_b
            latent_list.append(interp.astype(np.float32))
            dataset_indices.append(dataset_idx)
            alpha_list.append(float(alpha))
            record = {
                'dataset_index': dataset_idx,
                'alpha': float(alpha),
                'obsid_a': obsid_a,
                'obsid_b': obsid_b,
                'star_a_params': star_a_params,
                'star_b_params': star_b_params,
            }
            if args.dataset == 'single':
                record['partner_dataset_index'] = partner_idx
            metadata_records.append(record)

    latents = np.stack(latent_list)
    dataset_indices = np.array(dataset_indices, dtype=np.int32)
    alphas = np.array(alpha_list, dtype=np.float32)

    out_npz = Path(args.output_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, latents=latents, dataset_indices=dataset_indices, alphas=alphas)

    out_meta = Path(args.output_metadata)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(out_meta, 'w') as fh:
        json.dump({'records': metadata_records}, fh, indent=2)

    print(f"Saved interpolations to {out_npz} ({latents.shape[0]} entries)")
    print(f"Metadata written to {out_meta}")


if __name__ == '__main__':
    main()
