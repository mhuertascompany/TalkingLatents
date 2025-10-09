"""Evaluate multimodal model on precomputed interpolated FM latents."""

import argparse
import json
import os
from pathlib import Path

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
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum sequence length expected by the base LLM')
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


def load_interpolations(npz_path: Path, metadata_path: Path):
    data = np.load(npz_path)
    latents = data['latents']
    dataset_indices = data['dataset_indices']
    alphas = data['alphas']
    with open(metadata_path, 'r') as fh:
        meta_records = json.load(fh)['records']
    return latents, dataset_indices, alphas, meta_records


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


def encode_prompt(tokenizer: Tokenizer) -> torch.Tensor:
    prompt = "Describe the physical properties of this star."
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    return prompt, torch.tensor([tokens], dtype=torch.long)


def main():
    args = parse_args()
    device = setup_single_gpu(args.device)

    latents, dataset_indices, alphas, metadata_records = load_interpolations(
        Path(args.interpolation_npz), Path(args.metadata_json))

    latent_dim = latents.shape[1]

    model_path, tokenizer_path = get_model_path(args)
    tokenizer = Tokenizer(model_path=str(tokenizer_path))

    model = build_model(args, latent_dim, device)

    rng = np.random.default_rng(args.random_seed)
    total = latents.shape[0]
    num_eval = min(args.num_samples, total)
    eval_indices = rng.choice(total, size=num_eval, replace=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as fh:
        for i in eval_indices:
            latent = torch.from_numpy(latents[i]).unsqueeze(0).to(device)
            alpha = float(alphas[i])
            meta = metadata_records[i]

            prompt_text, input_ids = encode_prompt(tokenizer)
            input_ids = input_ids.to(device)
            target_ids = torch.full_like(input_ids, -100)
            feature_start_indices = torch.tensor([0], dtype=torch.long, device=device)
            answer_start_indices = torch.tensor([input_ids.size(1)], dtype=torch.long, device=device)

            batch = {
                'input_ids': input_ids,
                'target_ids': target_ids,
                'feature_start_indices': feature_start_indices,
                'answer_start_indices': answer_start_indices,
                'masked_spectra': latent,
                'input_texts': [prompt_text],
                'target_texts': [''],
            }

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
                'dataset_index': int(dataset_indices[i]),
                'alpha': alpha,
                'question': prompt_text,
                'true_answer': target_text,
                'generated_text': gen_text,
                'log_probs': logps,
                'metadata': meta,
            }
            fh.write(json.dumps(record) + '\n')

            del batch, latent, input_ids
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print(f"Wrote evaluation results to {output_path}")


if __name__ == '__main__':
    main()
