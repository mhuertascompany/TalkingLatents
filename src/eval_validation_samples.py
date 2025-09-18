import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.distributed as dist

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from llama3.llama.model import Transformer, ModelArgs
from nn.llm import MultimodalLlamaModel
from nn.train import LLMTrainer
from data.dataset_interpert import create_stellar_dataloaders
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor


def setup_single():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for evaluation")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        import fairscale.nn.model_parallel.initialize as fs_init
        if not fs_init.model_parallel_is_initialized():
            fs_init.initialize_model_parallel(1)
    except Exception:
        pass
    torch.cuda.set_device(0)
    return 0


def get_model_path(args):
    def ok(p: str) -> bool:
        return os.path.isfile(os.path.join(p, 'params.json')) and os.path.isfile(os.path.join(p, 'tokenizer.model'))
    if args.llm_path and ok(args.llm_path):
        return args.llm_path, os.path.join(args.llm_path, 'tokenizer.model')
    if args.llm_root and args.llm_model:
        c1 = os.path.join(args.llm_root, args.llm_model)
        if ok(c1):
            return c1, os.path.join(c1, 'tokenizer.model')
        c2 = os.path.join(c1, 'original')
        if ok(c2):
            return c2, os.path.join(c2, 'tokenizer.model')
    raise FileNotFoundError("Could not resolve LLaMA model directory — check --llm_path or --llm_root/--llm_model")


def load_llm(args) -> Transformer:
    model_dir, _ = get_model_path(args)
    with open(Path(model_dir) / 'params.json', 'r') as f:
        params = json.load(f)
    margs = ModelArgs(max_batch_size=args.batch_size, max_seq_len=args.max_seq_length, **params)
    model = Transformer(margs)
    # load weights if available
    ckpts = sorted(Path(model_dir).glob('*.pth'))
    if ckpts:
        state = torch.load(ckpts[0], map_location='cpu')
        if isinstance(state, dict) and ('model' in state or 'state_dict' in state):
            state = state.get('model', state.get('state_dict'))
        model.load_state_dict(state, strict=False)
    return model


def parse_args():
    p = argparse.ArgumentParser("Evaluate validation samples with a trained model")
    p.add_argument('--json_file', type=str, required=True)
    p.add_argument('--features_file', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='logs/eval')
    p.add_argument('--exp_name', type=str, default='eval_samples')
    p.add_argument('--llm_root', type=str, default=os.environ.get('LLM_ROOT', '/data/.llama'))
    p.add_argument('--llm_model', type=str, default='Llama3.1-8B')
    p.add_argument('--llm_path', type=str, default=None)
    p.add_argument('--weights', type=str, default=None, help='Path to plain model weights (.pth) for warm-start')
    p.add_argument('--resume_path', type=str, default=None, help='Composite checkpoint to load model state from')
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--max_seq_length', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--num_samples', type=int, default=3)
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--top_p', type=float, default=0.8)
    p.add_argument('--max_new_tokens', type=int, default=50)
    p.add_argument('--llm_precision', type=str, default='fp16', choices=['fp32','fp16','bf16'])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = setup_single()

    # Data
    model_dir, tokenizer_path = get_model_path(args)
    spectral_features = None
    if args.features_file and os.path.exists(args.features_file):
        import numpy as np
        spectral_features = np.load(args.features_file)
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])
    train_loader, val_loader, test_loader = create_stellar_dataloaders(
        json_file=args.json_file,
        features_array=spectral_features,
        spectral_transforms=transf,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42,
        num_spectral_features=1,
        cache_dir=os.path.join(args.output_dir, 'cache'),
        tokenizer_path=tokenizer_path,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    tokenizer = val_loader.dataset.tokenizer

    # Model
    llm = load_llm(args)
    if args.llm_precision == 'fp16':
        llm.half()
    elif args.llm_precision == 'bf16' and torch.cuda.is_bf16_supported():
        llm.to(dtype=torch.bfloat16)
    llm = llm.to(device)
    model = MultimodalLlamaModel(
        base_model=llm,
        fm_model=None,
        latent_dim=2048,
        hidden_dim=512,
        num_spectral_features=1,
    ).to(device)

    # Load provided weights/composite checkpoint if any
    if args.resume_path and os.path.isfile(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location='cpu')
        sd = ckpt.get('model', ckpt)
        if 'state_dict' in sd:
            sd = sd['state_dict']
        # strip module.
        new_sd = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith('module.') else k
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
    elif args.weights and os.path.isfile(args.weights):
        sd = torch.load(args.weights, map_location='cpu')
        if isinstance(sd, dict) and ('model' in sd or 'state_dict' in sd):
            sd = sd.get('model', sd.get('state_dict'))
        model.load_state_dict(sd, strict=False)

    # Trainer for evaluation helper
    # Prepare lora_params (merge project config if present)
    default_lora = {
        'freeze_strategy': 'none',      # no LoRA changes needed for eval
        'lora_start_epoch': 0,
        'lora_lr_multiplier': 1.0,
        'lora_rank': 8,
        'lora_alpha': 8.0,
        'lora_dropout': 0.0,
        'lora_target_modules': []
    }
    try:
        cfg_path = os.path.join(ROOT_DIR, 'src', 'llm_config.json')
        if os.path.isfile(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            if 'lora_params' in cfg and isinstance(cfg['lora_params'], dict):
                default_lora.update(cfg['lora_params'])
    except Exception:
        pass

    trainer = LLMTrainer(
        model=model,
        optimizer=None,
        criterion=None,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        world_size=1,
        log_path=args.output_dir,
        exp_name=args.exp_name,
        max_iter=-1,
        scheduler=None,
        lora_params=default_lora,
        scaler=None,
        use_amp=(args.llm_precision in ['fp16','bf16']),
        max_grad_norm=0.0,
        tokenizer=tokenizer,
    )

    # Monkey-patch generation settings for evaluate_validation_samples
    # by temporarily overriding defaults in the method call via kwargs is not supported,
    # so we rely on the values hardcoded in trainer (already set to low temp in recent patches).

    # Run evaluation samples
    print(f"Evaluating {args.num_samples} validation samples...")
    trainer.evaluate_validation_samples(device, epoch=0, num_samples=args.num_samples)
    out_file = os.path.join(args.output_dir, f"{args.exp_name}_validation_samples.json")
    print(f"Saved samples to: {out_file}")


if __name__ == '__main__':
    main()
