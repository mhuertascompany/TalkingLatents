"""Interpolate precomputed FM latents between Star A and Star B and generate outputs."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.dataset_comparative import create_comparative_dataloaders, collate_comparative_fn
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from src.simple_questions import _load_llm_model, get_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Interpolate FM latent between Star A and Star B")
    parser.add_argument('--json_file', type=str, required=True,
                        help='Path to single-star questions JSON (used for tokenizer only)')
    parser.add_argument('--comparative_json_file', type=str, default=None,
                        help='Path to comparative questions JSON (defaults to --json_file if omitted)')
    parser.add_argument('--features_file', type=str, required=True,
                        help='NumPy .npy file with precomputed FM embeddings (same as training)')
    parser.add_argument('--llm_path', type=str, default=None,
                        help='Optional explicit path to LLaMA weights')
    parser.add_argument('--split_cache_root', type=str, required=True,
                        help='Root directory with cached splits (created during training)')
    parser.add_argument('--resume_path', type=str, required=True,
                        help='Composite checkpoint for the trained multimodal model')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file to write results')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of comparative samples to evaluate')
    parser.add_argument('--alphas', type=str, default='-1.0,-0.5,0.0,0.5,1.0',
                        help='Comma-separated α values for interpolation')
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--num_spectral_features', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_features(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array of embeddings, got shape {arr.shape}")
    return arr


def load_comparative_dataset(args: argparse.Namespace, tokenizer_path: str):
    comp_json = args.comparative_json_file or args.json_file
    transforms = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    train_dl, val_dl, test_dl = create_comparative_dataloaders(
        json_file=comp_json,
        features_array=None,
        batch_size=args.batch_size,
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


def gather_latent(features: np.ndarray, indices: torch.Tensor) -> torch.Tensor:
    if indices.dim() == 0:
        idx = int(indices.item())
        return torch.from_numpy(features[idx]).unsqueeze(0)
    idxs = indices.cpu().numpy().astype(int)
    return torch.from_numpy(features[idxs])


def build_model(args: argparse.Namespace, latent_dim: int, device: torch.device) -> MultimodalLlamaModelMultiTokens:
    llm = _load_llm_model(args)
    llm = llm.to(device)
    model = MultimodalLlamaModelMultiTokens(
        base_model=llm,
        fm_model=None,
        latent_dim=latent_dim,
        hidden_dim=512,
        num_spectral_features=args.num_spectral_features,
        mode='two_star'
    ).to(device)

    ckpt = torch.load(args.resume_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    clean_state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    model.mode = 'two_star'
    return model


def main():
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = torch.device(args.device)

    features = load_features(Path(args.features_file))
    latent_dim = features.shape[1]

    model_path, tokenizer_path = get_model_path(args)
    loader = load_comparative_dataset(args, tokenizer_path)
    tokenizer = loader.dataset.tokenizer

    model = build_model(args, latent_dim, device)

    alphas = [float(x) for x in args.alphas.split(',') if x.strip()]
    if 0.0 not in alphas:
        alphas.append(0.0)
    alphas = sorted(alphas)

    selected_indices = random.sample(range(len(loader.dataset)),
                                     min(args.num_samples, len(loader.dataset)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as fh:
        for dataset_idx in selected_indices:
            sample = loader.dataset[dataset_idx]
            batch = collate_comparative_fn([sample])

            idx_a = batch['features_a_indices'][0] if 'features_a_indices' in batch else batch['star_a_df_index'][0]
            idx_b = batch['features_b_indices'][0] if 'features_b_indices' in batch else batch['star_b_df_index'][0]

            latent_a = gather_latent(features, idx_a).to(device)
            latent_b = gather_latent(features, idx_b).to(device)

            obsid_info = batch.get('obsid', [None])[0]
            if isinstance(obsid_info, (list, tuple)):
                obsid_a, obsid_b = obsid_info
            else:
                obsid_a, obsid_b = obsid_info, None

            stellar_a = batch.get('star_a_params', [{}])[0]
            stellar_b = batch.get('star_b_params', [{}])[0]

            for alpha in alphas:
                latent_interp = latent_a + alpha * latent_b
                batch_copy = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
                batch_copy['masked_spectra_a'] = latent_interp
                batch_copy['masked_spectra_b'] = latent_b

                gen_text, input_text, target_text, logps = model.generate_response_from_batch(
                    batch_data=batch_copy,
                    batch_idx=0,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                record = {
                    'dataset_index': dataset_idx,
                    'alpha': alpha,
                    'obsid_a': obsid_a,
                    'obsid_b': obsid_b,
                    'star_a_params': stellar_a,
                    'star_b_params': stellar_b,
                    'question': input_text,
                    'true_answer': target_text,
                    'generated_text': gen_text,
                    'log_probs': logps,
                }
                fh.write(json.dumps(record) + '\n')

    print(f"Wrote interpolated generations to {output_path}")


if __name__ == '__main__':
    main()
