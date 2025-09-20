import os
import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from data.dataset_interpert import create_stellar_dataloaders, collate_fn
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from src.simple_questions import _load_llm_model, _load_spectra_model


def parse_args():
    p = argparse.ArgumentParser("Evaluate multi-token model on random validation samples")
    p.add_argument('--json_file', type=str, required=True)
    p.add_argument('--features_file', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='logs/eval_multitok')
    p.add_argument('--exp_name', type=str, default='eval_multitok')
    p.add_argument('--llm_root', type=str, default=os.environ.get('LLM_ROOT', '/data/.llama'))
    p.add_argument('--llm_model', type=str, default='Llama3.1-8B')
    p.add_argument('--llm_path', type=str, default=None)
    p.add_argument('--resume_path', type=str, default=None, help='Composite checkpoint (model state)')
    p.add_argument('--weights', type=str, default=None, help='Plain weights .pth (optional)')
    p.add_argument('--llm_precision', type=str, default='fp16', choices=['fp32','fp16','bf16'])
    p.add_argument('--spectral_embedding_dim', type=int, default=2048)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--num_spectral_features', type=int, default=4)
    p.add_argument('--max_seq_length', type=int, default=96)
    p.add_argument('--num_samples', type=int, default=5)
    p.add_argument('--max_new_tokens', type=int, default=100)
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--top_p', type=float, default=0.8)
    p.add_argument('--random_seed', type=int, default=42)
    return p.parse_args()


def build_model(args, device):
    llm = _load_llm_model(args)
    if args.llm_precision == 'fp16':
        llm.half()
    elif args.llm_precision == 'bf16' and torch.cuda.is_bf16_supported():
        llm.to(dtype=torch.bfloat16)
    llm = llm.to(device)
    fm = None if args.features_file else _load_spectra_model()
    if fm is not None:
        fm = fm.to(device)
    model = MultimodalLlamaModelMultiTokens(
        base_model=llm,
        fm_model=fm,
        latent_dim=args.spectral_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_spectral_features=args.num_spectral_features,
        use_checkpoint=False,
    ).to(device)

    # Load composite resume or plain weights
    if args.resume_path and os.path.isfile(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location='cpu')
        sd = ckpt.get('model', ckpt)
        if 'state_dict' in sd:
            sd = sd['state_dict']
        new_sd = { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }
        model.load_state_dict(new_sd, strict=False)
    elif args.weights and os.path.isfile(args.weights):
        sd = torch.load(args.weights, map_location='cpu')
        if isinstance(sd, dict) and ('model' in sd or 'state_dict' in sd):
            sd = sd.get('model', sd.get('state_dict'))
        model.load_state_dict(sd, strict=False)
    return model


def compute_tf_perplexity(model, batch, device):
    input_ids = batch['input_ids'].to(device)
    target_ids = batch['target_ids'].to(device)
    input_spectra = batch['masked_spectra'].to(device)
    feat_start = batch['feature_start_indices'].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids,
                    input_spectra=input_spectra,
                    special_token_positions=feat_start)
        logits = out['logits']
        # Shift
        shift_logits = logits[..., -1 - (logits.size(1)-1):-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        mask = (shift_labels != -100)
        if not mask.any():
            return float('inf')
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = mask.view(-1)
        answer_logits = flat_logits[flat_mask]
        answer_labels = flat_labels[flat_mask]
        if answer_logits.numel() == 0:
            return float('inf')
        loss = torch.nn.functional.cross_entropy(answer_logits, answer_labels, reduction='mean')
        return torch.exp(loss).item()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

    # Build dataloaders with batch_size=1 for simple per-sample selection
    spectral_features = None
    if args.features_file and os.path.exists(args.features_file):
        spectral_features = np.load(args.features_file)
    from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    # Reuse helper but pass batch_size=1; we will sample directly from val_dataset
    _, val_loader, _ = create_stellar_dataloaders(
        json_file=args.json_file,
        features_array=spectral_features,
        spectral_transforms=transf,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=args.random_seed,
        num_spectral_features=args.num_spectral_features,
        cache_dir=os.path.join(args.output_dir, 'cache'),
        tokenizer_path=None,
        max_length=args.max_seq_length,
        batch_size=1,
        num_workers=0,
    )
    val_dataset = val_loader.dataset
    tokenizer = getattr(val_dataset, 'tokenizer', None)

    # Pick random indices
    total = len(val_dataset)
    picks = random.sample(range(total), k=min(args.num_samples, total))

    # Build model
    model = build_model(args, device)
    model.eval()

    results = {
        'epoch': 0,
        'samples': [],
        'avg_teacher_forcing_perplexity': 0.0,
        'avg_generation_perplexity': 0.0,
    }
    tf_sum = 0.0
    gen_sum = 0.0
    tf_cnt = 0
    gen_cnt = 0

    for idx in picks:
        sample = val_dataset[idx]
        batch = collate_fn([sample])

        tf_perp = compute_tf_perplexity(model, batch, device)

        gen_text, in_text, tgt_text, logps = model.generate_response_from_batch(
            batch_data=batch,
            batch_idx=0,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if logps:
            avg_logp = float(np.mean(logps))
            gen_perp = float(np.exp(-avg_logp))
        else:
            gen_perp = float('inf')

        results['samples'].append({
            'obsid': sample.get('obsid', 'Unknown') if isinstance(sample, dict) else 'Unknown',
            'question': in_text,
            'true_answer': tgt_text,
            'generated_answer': gen_text,
            'teacher_forcing_perplexity': None if np.isinf(tf_perp) else tf_perp,
            'generation_perplexity': None if np.isinf(gen_perp) else gen_perp,
            'num_generated_tokens': len(logps),
        })

        if not np.isinf(tf_perp):
            tf_sum += tf_perp
            tf_cnt += 1
        if not np.isinf(gen_perp):
            gen_sum += gen_perp
            gen_cnt += 1

    results['avg_teacher_forcing_perplexity'] = (tf_sum / tf_cnt) if tf_cnt > 0 else None
    results['avg_generation_perplexity'] = (gen_sum / gen_cnt) if gen_cnt > 0 else None

    out_file = os.path.join(args.output_dir, f"{args.exp_name}_validation_samples.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved samples to: {out_file}")


if __name__ == '__main__':
    main()

