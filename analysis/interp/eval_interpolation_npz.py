"""Evaluate multimodal model on precomputed interpolated FM latents."""

import argparse
import json
import os
import socket
import re
from pathlib import Path
from typing import Callable, Tuple, List

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from llama3.llama.tokenizer import Tokenizer
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from src.simple_questions import _load_llm_model, get_model_path
from data.dataset_interpert import create_stellar_dataloaders, collate_fn as single_collate_fn
from data.dataset_comparative import create_comparative_dataloaders, collate_comparative_fn
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor


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
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Dummy batch size needed by _load_llm_model')
    parser.add_argument('--num_spectral_features', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=25,
                        help='Number of interpolated latents to evaluate')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--features_file', type=str, default=None,
                        help='Optional features.npy used during training/evaluation')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum sequence length expected by the base LLM')
    parser.add_argument('--dataset', choices=['single', 'comparative'], default='comparative')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--alpha_sweep', action='store_true',
                        help='Evaluate a sweep of alpha values instead of using stored latents')
    parser.add_argument('--alpha_start', type=float, default=0.0)
    parser.add_argument('--alpha_end', type=float, default=1.0)
    parser.add_argument('--alpha_step', type=float, default=0.1)
    return parser.parse_args()


def _find_free_port(preferred: int = 29500) -> int:
    """Pick an available port, preferring the suggested one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('', preferred))
            return preferred
        except OSError:
            sock.bind(('', 0))
            return sock.getsockname()[1]


def setup_single_gpu(device_str: str) -> torch.device:
    if device_str != 'cuda':
        return torch.device(device_str)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    current_port = int(os.environ.get('MASTER_PORT', 29500))
    free_port = _find_free_port(current_port)
    os.environ['MASTER_PORT'] = str(free_port)
    if not dist.is_initialized():
        try:
            dist.init_process_group('gloo', rank=0, world_size=1)
        except Exception as exc:
            if 'Address already in use' in str(exc):
                os.environ['MASTER_PORT'] = str(_find_free_port())
                dist.init_process_group('gloo', rank=0, world_size=1)
            else:
                raise
    try:
        import fairscale.nn.model_parallel.initialize as fs_init
        if not fs_init.model_parallel_is_initialized():
            fs_init.initialize_model_parallel(1)
    except Exception:
        pass
    torch.cuda.set_device(0)
    return torch.device('cuda', 0)


def load_interpolations(npz_path: Path, metadata_path: Path):
    data = np.load(npz_path)
    latents = data['latents']
    dataset_indices = data['dataset_indices']
    alphas = data['alphas']
    with open(metadata_path, 'r') as fh:
        meta_records = json.load(fh)['records']
    return latents, dataset_indices, alphas, meta_records


_PHYS_PROP_SUBSTRINGS = (
    'teff', 'logg', 'feh', 'age', 'mass', 'radius', 'luminosity', 'stage', 'rv'
)


def extract_physical_properties(params):
    if not isinstance(params, dict):
        return None
    result = {}
    for key, value in params.items():
        key_lower = key.lower()
        if any(substr in key_lower for substr in _PHYS_PROP_SUBSTRINGS):
            result[key] = value
    return result or None


def _prefer_value(params: dict, candidates: List[str]):
    if not isinstance(params, dict):
        return None
    lowered = {k.lower(): k for k in params.keys()}
    for cand in candidates:
        if cand in lowered:
            return params[lowered[cand]]
    for key in params.keys():
        low = key.lower()
        if any(cand in low for cand in candidates):
            return params[key]
    return None


def extract_star_params(params: dict):
    teff = _prefer_value(params, ['teff', 'teff_k', 'teff_model_k'])
    logg = _prefer_value(params, ['logg', 'logg_model'])
    feh = _prefer_value(params, ['feh', '[fe/h]'])
    return teff, logg, feh


def parse_inferred_properties(text: str):
    teff_match = re.search(r'teff\s*[:=]\s*([-+]?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    logg_match = re.search(r'log\s*g?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    feh_match = re.search(r'(feh|\[fe/?h\])\s*[:=]\s*([-+]?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    teff = float(teff_match.group(1)) if teff_match else None
    logg = float(logg_match.group(1)) if logg_match else None
    feh = float(feh_match.group(2)) if feh_match else None
    return teff, logg, feh


def to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_model(args: argparse.Namespace, latent_dim: int, device: torch.device) -> MultimodalLlamaModelMultiTokens:
    llm = _load_llm_model(args)
    llm = llm.to(device)
    model = MultimodalLlamaModelMultiTokens(
        base_model=llm,
        fm_model=None,
        latent_dim=latent_dim,
        hidden_dim=512,
        num_spectral_features=args.num_spectral_features,
        mode='single_star'
    ).to(device)
    ckpt = torch.load(args.resume_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    clean_state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    model.mode = 'single_star'
    return model


def load_eval_dataset(args: argparse.Namespace,
                      tokenizer_path: str,
                      features: np.ndarray | None) -> Tuple[torch.utils.data.Dataset, Callable]:
    transforms = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    split_root = Path(args.split_cache_root)
    split_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'single':
        cache_dir = split_root / 'single' / Path(args.json_file).stem
        train_dl, val_dl, test_dl = create_stellar_dataloaders(
            json_file=args.json_file,
            features_array=features,
            spectral_transforms=transforms,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_seed,
            num_spectral_features=args.num_spectral_features,
            cache_dir=str(cache_dir),
            allow_new_splits=False,
            tokenizer_path=tokenizer_path,
            max_length=args.max_seq_length,
            batch_size=1,
            num_workers=args.num_workers,
        )
        loaders = {'train': train_dl, 'val': val_dl, 'test': test_dl}
        return loaders[args.split].dataset, single_collate_fn

    comp_json = args.comparative_json_file or args.json_file
    cache_dir = split_root / 'comparative' / Path(comp_json).stem
    train_dl, val_dl, test_dl = create_comparative_dataloaders(
        json_file=comp_json,
        features_array=features,
        batch_size=1,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        num_workers=args.num_workers,
        cache_dir=str(cache_dir),
        tokenizer_path=tokenizer_path,
        max_length=args.max_seq_length,
        num_stellar_features=args.num_spectral_features,
        allow_new_splits=False,
    )
    loaders = {'train': train_dl, 'val': val_dl, 'test': test_dl}
    return loaders[args.split].dataset, collate_comparative_fn


def build_prompt_batch(tokenizer: Tokenizer, prompt_text: str, device: torch.device,
                       max_seq_length: int) -> dict:
    tokens = tokenizer.encode(prompt_text, bos=True, eos=False)
    tokens = tokens[:max_seq_length]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    target_ids = torch.full_like(input_ids, -100)
    feature_start_indices = torch.tensor([0], dtype=torch.long, device=device)
    answer_start_indices = torch.tensor([input_ids.size(1)], dtype=torch.long, device=device)
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'feature_start_indices': feature_start_indices,
        'answer_start_indices': answer_start_indices,
        'input_texts': [prompt_text],
        'target_texts': [''],
    }


def main():
    args = parse_args()
    device = setup_single_gpu(args.device)

    latents, dataset_indices, alphas, metadata_records = load_interpolations(
        Path(args.interpolation_npz), Path(args.metadata_json))

    latent_dim = latents.shape[1]

    model_path, tokenizer_path = get_model_path(args)
    tokenizer = Tokenizer(model_path=str(tokenizer_path))

    model = build_model(args, latent_dim, device)
    print(f"Loaded MultimodalLlamaModelMultiTokens from {args.resume_path}")

    rng = np.random.default_rng(args.random_seed)
    total = latents.shape[0]
    num_eval = min(args.num_samples, total)
    eval_indices = rng.choice(total, size=num_eval, replace=False)

    features_array = None
    if args.features_file and os.path.exists(args.features_file):
        features_array = np.load(args.features_file)

    dataset, collate_fn = load_eval_dataset(args, tokenizer_path, features_array)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sweep_alphas = None
    if args.alpha_sweep:
        if args.alpha_step <= 0:
            raise ValueError('alpha_step must be positive when using alpha_sweep')
        sweep_alphas = np.arange(args.alpha_start, args.alpha_end + 1e-8, args.alpha_step)
        if sweep_alphas.size == 0:
            raise ValueError('No alpha values generated; check alpha_start/end/step settings')

    seen_keys = set()
    selected_indices: List[int] = []
    for idx in eval_indices:
        meta = metadata_records[idx]
        ds_idx = int(dataset_indices[idx])
        partner_idx = meta.get('partner_dataset_index') if args.dataset == 'single' else meta.get('obsid_b')
        key = (ds_idx, partner_idx)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected_indices.append(idx)
        if len(selected_indices) >= args.num_samples:
            break

    with open(output_path, 'w') as fh:
        if args.alpha_sweep:
            prompt_template = (
                "This latent embedding corresponds to a star obtained by interpolating between two "
                "stars with weight alpha={alpha:.2f}. Estimate its physical properties and respond exactly "
                "in the format: Teff=<value> K, logg=<value>, FeH=<value>."
            )

            for idx in selected_indices:
                meta = metadata_records[idx]
                dataset_idx = int(dataset_indices[idx])

                if args.dataset == 'single':
                    partner_idx = meta.get('partner_dataset_index')
                    if partner_idx is None:
                        raise ValueError('partner_dataset_index missing in metadata for single dataset sweep')
                    sample_a = dataset[dataset_idx]
                    sample_b = dataset[partner_idx]
                    batch_a = single_collate_fn([sample_a])
                    batch_b = single_collate_fn([sample_b])
                    latent_a = batch_a['masked_spectra'][0].cpu().numpy().astype(np.float32)
                    latent_b = batch_b['masked_spectra'][0].cpu().numpy().astype(np.float32)
                else:
                    sample = dataset[dataset_idx]
                    base_batch = collate_fn([sample])
                    latent_a = base_batch['masked_spectra_a'][0].cpu().numpy().astype(np.float32)
                    latent_b = base_batch['masked_spectra_b'][0].cpu().numpy().astype(np.float32)
                    partner_idx = None

                star_a_params = meta.get('star_a_params', {}) or {}
                star_b_params = meta.get('star_b_params', {}) or {}
                teff_a, logg_a, feh_a = extract_star_params(star_a_params)
                teff_b, logg_b, feh_b = extract_star_params(star_b_params)
                teff_a, logg_a, feh_a = to_float(teff_a), to_float(logg_a), to_float(feh_a)
                teff_b, logg_b, feh_b = to_float(teff_b), to_float(logg_b), to_float(feh_b)

                fh.write(
                    f"# dataset_index={dataset_idx}, partner_index={partner_idx}, "
                    f"obsid_a={meta.get('obsid_a')}, obsid_b={meta.get('obsid_b')}\n"
                )
                fh.write(
                    "alpha,inferred_teff,inferred_logg,inferred_feh,"
                    "star_a_teff,star_a_logg,star_a_feh,"
                    "star_b_teff,star_b_logg,star_b_feh\n"
                )

                for alpha in sweep_alphas:
                    interp = (1.0 - alpha) * latent_a + alpha * latent_b
                    latent_tensor = torch.from_numpy(interp.astype(np.float32)).unsqueeze(0).to(device)

                    prompt_text = prompt_template.format(alpha=alpha)
                    batch_prompt = build_prompt_batch(tokenizer, prompt_text, device, args.max_seq_length)
                    batch_prompt['masked_spectra'] = latent_tensor

                    with torch.no_grad():
                        gen_text, _, _, _ = model.generate_response_from_batch(
                            batch_data=batch_prompt,
                            batch_idx=0,
                            tokenizer=tokenizer,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )

                    pred_teff, pred_logg, pred_feh = parse_inferred_properties(gen_text)

                    def fmt(val):
                        if val is None:
                            return ''
                        if abs(val) >= 100:
                            return f"{val:.1f}"
                        return f"{val:.3f}"

                    row = [
                        f"{alpha:.2f}",
                        fmt(pred_teff),
                        fmt(pred_logg),
                        fmt(pred_feh),
                        fmt(teff_a),
                        fmt(logg_a),
                        fmt(feh_a),
                        fmt(teff_b),
                        fmt(logg_b),
                        fmt(feh_b),
                    ]
                    fh.write(','.join(row) + '\n')
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                fh.write('\n')
        else:
            for idx in selected_indices:
                dataset_idx = int(dataset_indices[idx])
                sample = dataset[dataset_idx]
                batch = collate_fn([sample])

                latent_np = latents[idx].astype(np.float32)
                alpha = float(alphas[idx])
                meta = metadata_records[idx]

                if args.dataset == 'single':
                    latent_tensor = torch.from_numpy(latent_np).to(batch['masked_spectra'].dtype)
                    batch['masked_spectra'][0] = latent_tensor
                    if batch.get('features') is not None and isinstance(batch['features'], torch.Tensor):
                        batch['features'][0] = latent_tensor
                else:
                    latent_tensor = torch.from_numpy(latent_np)
                    if 'masked_spectra_a' in batch:
                        batch['masked_spectra_a'][0] = latent_tensor

                with torch.no_grad():
                    gen_text, input_text, target_text, logps = model.generate_response_from_batch(
                        batch_data=batch,
                        batch_idx=0,
                        tokenizer=tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )

                record = {
                    'dataset_index': dataset_idx,
                    'alpha': alpha,
                    'question': input_text,
                    'true_answer': target_text,
                    'generated_text': gen_text,
                    'log_probs': logps,
                    'metadata': {
                        'obsid_a': meta.get('obsid_a'),
                        'obsid_b': meta.get('obsid_b'),
                        'star_a_physical_properties': extract_physical_properties(meta.get('star_a_params')),
                        'star_b_physical_properties': extract_physical_properties(meta.get('star_b_params')),
                        'partner_dataset_index': meta.get('partner_dataset_index'),
                    },
                }
                fh.write(json.dumps(record) + '\n')

                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    print(f"Wrote evaluation results to {output_path}")


if __name__ == '__main__':
    main()
