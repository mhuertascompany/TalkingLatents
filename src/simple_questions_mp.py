import os
import sys
import json
import yaml
import gc
import argparse
import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)

from llama3.llama.model import Transformer, ModelArgs
from nn.llm import MultimodalLlamaModel
from nn.train import LLMTrainer
from data.dataset_interpert import create_stellar_dataloaders
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor


def print_detailed_memory():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        available = total_memory - allocated
        print(f"GPU Memory Details:")
        print(f"  Total GPU Memory: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Max Allocated: {max_allocated:.2f} GB")
        print(f"  Available: {available:.2f} GB")
        print(f"  Free (Reserved - Allocated): {(reserved - allocated):.2f} GB")


def get_model_path(args):
    def is_valid_model_dir(p: str) -> bool:
        return os.path.isfile(os.path.join(p, 'params.json')) and os.path.isfile(os.path.join(p, 'tokenizer.model'))

    candidates = []
    if getattr(args, 'llm_path', None):
        candidates.append(args.llm_path)
    if getattr(args, 'llm_root', None) and getattr(args, 'llm_model', None):
        candidates.append(os.path.join(args.llm_root, args.llm_model))
    if getattr(args, 'llm_root', None):
        candidates.append(args.llm_root)
    if getattr(args, 'llm_root', None) and getattr(args, 'llm_model', None):
        candidates.append(os.path.join(args.llm_root, args.llm_model, 'original'))
    if getattr(args, 'llm_root', None):
        candidates.append(os.path.join(args, 'llm_root', 'original'))

    chosen = None
    for c in candidates:
        if c and is_valid_model_dir(c):
            chosen = c
            break
    if not chosen:
        raise FileNotFoundError(f"Could not resolve LLaMA model dir. Checked: {candidates}")
    print(f"Using LLaMA model directory: {chosen}")
    return chosen, os.path.join(chosen, 'tokenizer.model')


def _load_llm_model_with_error_handling(args) -> Transformer:
    model_path, _ = get_model_path(args)
    max_batch_size, max_seq_len = args.batch_size, args.max_seq_length
    with open(Path(model_path) / "params.json", "r") as f:
        params = json.loads(f.read())
    print(f"Model params from config: {params}")
    model_args = ModelArgs(max_batch_size=max_batch_size, max_seq_len=max_seq_len, **params)
    print("Creating LLaMA model on CPU...")
    model = Transformer(model_args)
    checkpoints = sorted(Path(model_path).glob("*.pth"))
    if checkpoints:
        print(f"Loading LLaMA checkpoint: {checkpoints[0]}")
        try:
            checkpoint = torch.load(checkpoints[0], map_location="cpu")
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
            model.load_state_dict(checkpoint, strict=False)
            del checkpoint
            gc.collect()
            print("✓ LLaMA model loaded successfully (partial weights ok)")
        except Exception as e:
            print(f"Error loading checkpoint: {e}\nProceeding with random init...")
    else:
        print("No checkpoints found, using randomly initialized model")
    return model


def parse_args():
    p = argparse.ArgumentParser("LLaMA MP training")
    # Data
    p.add_argument('--json_file', type=str, required=True)
    p.add_argument('--features_file', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='logs')
    p.add_argument('--exp_name', type=str, default='interpert')
    # LLM
    p.add_argument('--llm_root', type=str, default=os.environ.get('LLM_ROOT', '/data/.llama'))
    p.add_argument('--llm_model', type=str, default='Llama3.1-8B')
    p.add_argument('--llm_path', type=str, default=None)
    p.add_argument('--llm_precision', type=str, default='fp16', choices=['fp32','fp16','bf16'])
    p.add_argument('--spectral_embedding_dim', type=int, default=2048)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--num_spectral_features', type=int, default=1)
    p.add_argument('--latent_ids', type=list, nargs='*', default=['Teff', 'logg', 'FeH'])
    p.add_argument('--max_seq_length', type=int, default=128)
    p.add_argument('--checkpoint_dir', type=str, default=None)
    p.add_argument('--train', type=bool, default=True)
    # Train
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--num_epochs', type=int, default=1)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--warmup_epochs', type=int, default=2)
    p.add_argument('--early_stopping', type=int, default=20)
    p.add_argument('--max_iter', type=int, default=2000)
    p.add_argument('--use_amp', action='store_true', default=False)
    p.add_argument('--gradient_checkpointing', action='store_true', default=True)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    # Splits
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--test_ratio', type=float, default=0.1)
    p.add_argument('--random_seed', type=int, default=42)
    # Loader
    p.add_argument('--num_workers', type=int, default=0)
    # Freezing
    p.add_argument('--freeze_llm', action='store_true', default=True)
    p.add_argument('--freeze_spectral', action='store_true', default=True)
    # Eval cadence
    p.add_argument('--eval_every', type=int, default=1)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--compute_retrieval_metrics', action='store_true')
    # Model parallel size
    p.add_argument('--mp_size', type=int, default=None, help='Tensor model-parallel size (defaults to WORLD_SIZE)')
    return p.parse_args()


def setup_mp(mp_size: Optional[int] = None):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    jobid = int(os.environ.get("SLURM_JOBID", 0))
    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
    default_port = 12910 + (jobid % 20000) if jobid else 12910
    os.environ.setdefault("MASTER_PORT", str(default_port))

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")
    torch.cuda.set_device(local_rank)

    # Initialize FairScale model parallel with mp_size = world_size or user-provided
    if mp_size is None:
        mp_size = world_size
    print(f"> initializing model parallel with size {mp_size}")
    import fairscale.nn.model_parallel.initialize as fs_init
    if not fs_init.model_parallel_is_initialized():
        fs_init.initialize_model_parallel(mp_size)
    return local_rank, world_size, mp_size


def create_datasets_and_loaders_mp(args, device):
    spectral_features = None
    if args.features_file and os.path.exists(args.features_file):
        print(f"Loading spectral features from {args.features_file}")
        import numpy as np
        spectral_features = np.load(args.features_file)
        print(f"Spectral features shape: {spectral_features.shape}")

    _, tokenizer_path = get_model_path(args)
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])

    cache_dir_base = os.path.join(args.output_dir, 'cache')
    rank = dist.get_rank() if dist.is_initialized() else 0
    cache_dir = cache_dir_base if rank == 0 else f"{cache_dir_base}_r{rank}"
    os.makedirs(cache_dir, exist_ok=True)

    # Temporarily disable DistributedSampler in helper (we use only tensor model parallel)
    import torch.distributed as dist_mod
    saved_is_init = dist_mod.is_initialized
    dist_mod.is_initialized = lambda: False
    try:
        train_loader, val_loader, test_loader = create_stellar_dataloaders(
            json_file=args.json_file,
            features_array=spectral_features,
            spectral_transforms=transf,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_seed,
            num_spectral_features=args.num_spectral_features,
            cache_dir=cache_dir,
            tokenizer_path=tokenizer_path,
            max_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    finally:
        dist_mod.is_initialized = saved_is_init
    return train_loader, val_loader, test_loader


def create_model_mp(args, device):
    print_detailed_memory()
    llm_model = _load_llm_model_with_error_handling(args)
    if args.llm_precision == 'fp16':
        llm_model.half(); print("✓ LLM weights cast to float16")
    elif args.llm_precision == 'bf16' and torch.cuda.is_bf16_supported():
        llm_model.to(dtype=torch.bfloat16); print("✓ LLM weights cast to bfloat16")
    if args.gradient_checkpointing and hasattr(llm_model, 'gradient_checkpointing_enable'):
        llm_model.gradient_checkpointing_enable(); print("✓ Gradient checkpointing enabled for LLM")
    llm_model = llm_model.to(device)
    print("=== After LLM Model ==="); print_detailed_memory()

    spectral_model = None
    model = MultimodalLlamaModel(
        base_model=llm_model,
        fm_model=spectral_model,
        latent_dim=args.spectral_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_spectral_features=args.num_spectral_features,
    ).to(device)

    if args.freeze_llm and hasattr(model, 'base_model') and model.base_model is not None:
        for p in model.base_model.parameters():
            p.requires_grad = False
        print("✓ LLM base_model frozen (no grads)")
    print("=== Final Memory State ==="); print_detailed_memory()
    return model


def create_optimizer_and_scheduler(model, args, train_loader):
    named_params = list(model.named_parameters())
    for name, param in named_params:
        if 'base_model' in name and args.freeze_llm:
            param.requires_grad = False
    opt_params = [p for n, p in named_params if isinstance(p, torch.nn.Parameter) and p.requires_grad]
    if len(opt_params) == 0:
        raise RuntimeError("No trainable parameters found for optimizer.")
    optimizer = torch.optim.AdamW(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.num_epochs)
    warmup_steps = len(train_loader) * args.warmup_epochs
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / float(warmup_steps)
        import math
        denom = max(1, total_steps - max(0, warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * (step - max(0, warmup_steps)) / denom))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=args.use_amp)
    return optimizer, scheduler, scaler


def save_config(args, output_dir):
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Training configuration saved to {config_path}.")


def main():
    args = parse_args()
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    args.output_dir = os.path.join(args.output_dir, date)
    os.makedirs(args.output_dir, exist_ok=True)

    local_rank, world_size, mp_size = setup_mp(args.mp_size)
    print(f"MP setup complete. local_rank={local_rank}, world_size={world_size}, mp_size={mp_size}")

    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)

    print("Creating datasets and dataloaders (no data parallel sharding)...")
    train_loader, val_loader, test_loader = create_datasets_and_loaders_mp(args, local_rank)
    tokenizer = train_loader.dataset.tokenizer

    print("Creating model (tensor model parallel)...")
    model = create_model_mp(args, local_rank)

    print("Creating optimizer and scheduler...")
    optimizer, scheduler, scaler = create_optimizer_and_scheduler(model, args, train_loader)
    criterion = torch.nn.CrossEntropyLoss()

    print("Creating trainer...")
    with open(os.path.join(ROOT_DIR, "src", "llm_config.json"), "r") as f:
        config = json.load(f)
    lora_params = config['lora_params']
    trainer = LLMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=local_rank,
        world_size=world_size,
        log_path=args.output_dir,
        exp_name=args.exp_name,
        max_iter=args.max_iter,
        scheduler=scheduler,
        lora_params=lora_params,
        scaler=scaler,
        use_amp=args.use_amp,
        max_grad_norm=args.max_grad_norm,
        tokenizer=tokenizer,
    )

    if local_rank == 0:
        save_config(args, args.output_dir)

    if args.train:
        if local_rank == 0:
            trainer.evaluate_validation_samples(local_rank, 0)
        print("Starting training (MP only, no DDP)...")
        _ = trainer.fit(num_epochs=args.num_epochs, device=local_rank, early_stopping=args.early_stopping, best='loss')

    if local_rank == 0:
        print("Running final evaluation on test set...")
        try:
            test_results = trainer.predict(test_loader, local_rank, tokenizer=tokenizer)
        except TypeError:
            test_results = trainer.predict(test_loader, local_rank)
        print("Evaluation complete.")


def run_with_teardown():
    try:
        main()
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback; traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


if __name__ == "__main__":
    print("=" * 80)
    print("CLIP MULTIMODAL STELLAR MODEL TRAINING (Model Parallel)")
    print("=" * 80)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb=64')
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    run_with_teardown()

