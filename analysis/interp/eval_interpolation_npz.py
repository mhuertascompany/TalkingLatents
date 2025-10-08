"""Evaluate multimodal model on precomputed interpolated FM latents."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.dataset_comparative import create_comparative_dataloaders, collate_comparative_fn
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from src.simple_questions import _load_llm_model, get_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate interpolated FM latents")
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--comparative_json_file', type=str, default=None)
    parser.add_argument('--split_cache_root', type=str, required=True)
    parser.add_argument('--resume_path', type=str, required=True)
    parser.add_argument('--interpolation_npz', type=str, required=True)
    parser.add_argument('--metadata_json', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--llm_path', type=str, default=None,
                        help='Optional explicit path to LLaMA weights')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--num_spectral_features', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Dummy batch size needed by _load_llm_model')
    parser.add_argument('--num_samples', type=int, default=25,
                        help='Number of interpolations to evaluate (random subset)')
    return parser.parse_args()


def setup_single_gpu(device_str: str) -> torch.device:
    if device_str != 'cuda':
        return torch.device(device_str)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')
    if not dist.is_initialized():
        dist.init_process_group('gloo', rank=0, world_size=1)
    try:
        import fairscale.nn.model_parallel.initialize as fs_init
        if not fs_init.model_parallel_is_initialized():
            fs_init.initialize_model_parallel(1)
    except Exception:
        pass
    torch.cuda.set_device(0)
    return torch.device('cuda', 0)


def load_dataset(args: argparse.Namespace, tokenizer_path: str):
    comp_json = args.comparative_json_file or args.json_file
    transforms = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    train_dl, val_dl, test_dl = create_comparative_dataloaders(
        json_file=comp_json,
        features_array=None,
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
    return {'train': train_dl, 'val': val_dl, 'test': test_dl}[args.split]


def load_interpolations(npz_path: Path, metadata_path: Path):
    data = np.load(npz_path)
    latent_a = data['latent_a']
    latent_b = data['latent_b']
    dataset_indices = data['dataset_indices']
    alphas = data['alphas']
    with open(metadata_path, 'r') as fh:
        meta_records = json.load(fh)['records']
    return latent_a, latent_b, dataset_indices, alphas, meta_records


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


def make_gpu_batch(batch: Dict[str, Any], latent_a: torch.Tensor,
                   latent_b: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    keys = ['input_ids', 'target_ids', 'star_a_feature_indices', 'star_b_feature_indices',
            'answer_start_indices']
    gpu_batch = {k: batch[k].to(device) for k in keys if k in batch}
    gpu_batch['masked_spectra_a'] = latent_a
    gpu_batch['masked_spectra_b'] = latent_b
    return gpu_batch


def main():
    args = parse_args()
    device = setup_single_gpu(args.device)

    latent_a, latent_b, dataset_indices, alphas, meta_records = load_interpolations(
        Path(args.interpolation_npz), Path(args.metadata_json))

    latent_dim = latent_a.shape[1]

    model_path, tokenizer_path = get_model_path(args)
    loader = load_dataset(args, tokenizer_path)
    dataset = loader.dataset
    tokenizer = dataset.tokenizer

    model = build_model(args, latent_dim, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = latent_a.shape[0]
    rng = np.random.default_rng(args.random_seed)
    num_eval = min(args.num_samples, total)
    eval_indices = rng.choice(total, size=num_eval, replace=False)

    with open(output_path, 'w') as fh:
        for i in eval_indices:
            idx = int(dataset_indices[i])
            alpha = float(alphas[i])
            meta = meta_records[i]

            sample = dataset[idx]
            batch = collate_comparative_fn([sample])

            latent_a_tensor = torch.from_numpy(latent_a[i]).unsqueeze(0).to(device)
            latent_b_tensor = torch.from_numpy(latent_b[i]).unsqueeze(0).to(device)

            gpu_batch = make_gpu_batch(batch, latent_a_tensor, latent_b_tensor, device)

            with torch.no_grad():
                gen_text, input_text, target_text, logps = model.generate_response_from_batch(
                    batch_data=gpu_batch,
                    batch_idx=0,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

            record = {
                'dataset_index': idx,
                'alpha': alpha,
                'question': input_text,
                'true_answer': target_text,
                'generated_text': gen_text,
                'log_probs': logps,
                'metadata': meta,
            }
            fh.write(json.dumps(record) + '\n')

            del gpu_batch, latent_a_tensor, latent_b_tensor
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print(f"Wrote evaluation results to {output_path}")


if __name__ == '__main__':
    main()
