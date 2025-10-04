import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.dataset_interpert import StellarQuestionsDataset, collate_fn
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from src.simple_questions import _load_llm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply CAV perturbations to FM latents and generate descriptions.")
    parser.add_argument("--json-file", required=True, help="Path to stellar questions JSON file")
    parser.add_argument("--concept-file", required=True, help="Concept bundle JSON")
    parser.add_argument("--latents-npz", required=True, help="NPZ file produced by extract_fm_latents.py")
    parser.add_argument("--cav-file", required=True, help="NPZ file containing a concept vector (key 'vector')")
    parser.add_argument("--metadata-json", required=True, help="Metadata JSON from extract_fm_latents.py")
    parser.add_argument("--resume-path", required=True, help="Composite checkpoint for the multimodal model")
    parser.add_argument("--split-cache-root", required=True, help="Root directory with cached splits")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--concept", default=None, help="Concept name to evaluate (defaults to all in latents file)")
    parser.add_argument("--num-samples", type=int, default=10, help="Samples per concept")
    parser.add_argument("--alphas", type=str, default="-1.5,-0.5,0,0.5,1.5")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--num-spectral-features", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--llm-path", type=str, default=None)
    parser.add_argument("--llm-root", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default='Llama3.1-8B')
    parser.add_argument("--exp-name", type=str, default='cav_eval')
    parser.add_argument("--output-dir", type=str, default='logs/cav_eval')
    return parser.parse_args()


def load_latents(latent_path: Path) -> Dict[str, Any]:
    data = np.load(latent_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    obsids = data['obsids']
    concept_names = list(data['concept_names'])
    return {
        'embeddings': embeddings,
        'labels': labels,
        'obsids': obsids,
        'concept_names': concept_names,
    }



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


def build_model(args: argparse.Namespace, device: torch.device, latent_dim: int) -> MultimodalLlamaModelMultiTokens:
    llm = _load_llm_model(args)
    llm = llm.to(device)
    model = MultimodalLlamaModelMultiTokens(
        base_model=llm,
        fm_model=None,
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        num_spectral_features=args.num_spectral_features,
    ).to(device)
    # load checkpoint state
    ckpt = torch.load(args.resume_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    clean_state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")
    model.eval()
    model.mode = "single_star"
    return model



def map_index_to_embedding(latents: Dict[str, Any], metadata_path: Path) -> Dict[int, np.ndarray]:
    with open(metadata_path, 'r') as fh:
        meta = json.load(fh)
    index_to_embedding: Dict[int, np.ndarray] = {}
    embeddings = latents['embeddings']
    labels = latents['labels']
    concept_names = latents['concept_names']
    for record, emb, label in zip(meta['records'], embeddings, labels):
        index_to_embedding[int(record['dataset_index'])] = emb
    return index_to_embedding


def decode_tokens(tokenizer, ids: torch.Tensor) -> str:
    if tokenizer is None:
        return ''
    arr = ids.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[0]
    return tokenizer.decode(arr)


def generate_with_latent(model: MultimodalLlamaModelMultiTokens,
                          tokenizer,
                          batch: Dict[str, torch.Tensor],
                          latent: torch.Tensor,
                          max_new_tokens: int,
                          temperature: float,
                          top_p: float,
                          device: torch.device) -> Dict[str, Any]:
    input_ids = batch['input_ids'].to(device)
    feature_start_indices = batch['feature_start_indices'].to(device)
    answer_start_idx = batch['answer_start_indices'][0].item()

    prompt = input_ids[:, :max(1, answer_start_idx)].clone()
    generated_tokens: List[int] = []
    log_probs: List[float] = []

    def sample_top_p(logits: torch.Tensor) -> int:
        if temperature > 0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        if 0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cdf > top_p).float().argmax().item()
            cutoff = max(1, cutoff)
            sorted_probs = sorted_probs[:cutoff]
            sorted_idx = sorted_idx[:cutoff]
            sorted_probs = sorted_probs / sorted_probs.sum()
            next_idx = torch.multinomial(sorted_probs, 1).item()
            return sorted_idx[next_idx].item()
        else:
            return torch.multinomial(probs, 1).item()

    for _ in range(max_new_tokens):
        out = model._forward_no_cache(prompt, latent, feature_start_indices)
        logits = out['logits'][:, -1, :].squeeze(0)
        if temperature > 0:
            logits_scaled = logits / temperature
        else:
            logits_scaled = logits
        probs = torch.softmax(logits_scaled, dim=-1)
        next_token = sample_top_p(logits)
        generated_tokens.append(next_token)
        log_probs.append(torch.log(probs[next_token]).item())

        next_tensor = torch.tensor([[next_token]], device=device, dtype=prompt.dtype)
        prompt = torch.cat([prompt, next_tensor], dim=1)

        eos_id = getattr(tokenizer, 'eos_id', None)
        if eos_id is not None and next_token == eos_id:
            break

    generated_ids = torch.tensor(generated_tokens, dtype=input_ids.dtype).unsqueeze(0)
    text = decode_tokens(tokenizer, generated_ids)
    return {
        'generated_tokens': generated_tokens,
        'generated_text': text,
        'log_probs': log_probs,
    }


def main():
    args = parse_args()
    json_path = Path(args.json_file)
    latents_path = Path(args.latents_npz)
    metadata_path = Path(args.metadata_json)
    concept_path = Path(args.concept_file)
    cav_path = Path(args.cav_file)

    split_dir = resolve_split_cache(args, json_path)
    dataset = build_dataset(args, json_path, split_dir)
    tokenizer = getattr(dataset, 'tokenizer', None)

    latents = load_latents(latents_path)
    index_to_embedding = map_index_to_embedding(latents, metadata_path)

    cav_data = np.load(cav_path)
    cav_vector = cav_data['vector']
    cav_vector = cav_vector.astype(np.float32)
    norm = np.linalg.norm(cav_vector)
    if norm == 0:
        raise ValueError("CAV vector has zero norm")
    cav_vector = cav_vector / norm

    prepared_samples = []
    with open(concept_path, 'r') as fh:
        bundle = json.load(fh)
    for concept_name, entries in bundle['samples'].items():
        if args.concept and concept_name != args.concept:
            continue
        for entry in entries[:args.num_samples]:
            prepared_samples.append({
                'concept': concept_name,
                'dataset_index': int(entry['index'])
            })

    device = torch.device(args.device)
    latent_dim = latents['embeddings'].shape[1]
    model = build_model(args, device, latent_dim)

    alphas = [float(x) for x in args.alphas.split(',') if x.strip()]
    if 0.0 not in alphas:
        alphas.append(0.0)
    alphas = sorted(alphas)

    results = []

    for sample_info in prepared_samples:
        idx = sample_info['dataset_index']
        if idx not in index_to_embedding:
            print(f"Skipping sample {idx}: no embedding found")
            continue
        sample = dataset[idx]
        batch = collate_fn([sample])
        base_latent = index_to_embedding[idx]
        base_latent = torch.tensor(base_latent, dtype=torch.float32, device=device).unsqueeze(0)

        feature_results = []
        for alpha in alphas:
            perturb = torch.tensor(cav_vector * alpha, dtype=torch.float32, device=device).unsqueeze(0)
            latent = base_latent + perturb
            gen = generate_with_latent(model, tokenizer, batch, latent,
                                       max_new_tokens=args.max_new_tokens,
                                       temperature=args.temperature,
                                       top_p=args.top_p,
                                       device=device)
            feature_results.append({
                'alpha': alpha,
                'generated_text': gen['generated_text'],
                'log_probs': gen['log_probs'],
                'num_tokens': len(gen['generated_tokens']),
            })

        input_text = batch.get('input_texts', [''])[0]
        target_text = batch.get('target_texts', [''])[0]
        obsid = batch.get('obsids', [None])[0]

        results.append({
            'dataset_index': idx,
            'concept': sample_info['concept'],
            'obsid': obsid,
            'question': input_text,
            'true_answer': target_text,
            'results': feature_results,
        })

    with open(args.output, 'w') as out_f:
        for record in results:
            out_f.write(json.dumps(record) + '\n')

    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == '__main__':
    main()
