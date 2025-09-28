import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import os
os.system('pip install tiktoken fairscale fire blobfile')
from src.simple_questions import (
    setup,
    create_datasets_and_loaders,
    create_model_memory_optimized,
    save_config,
)
from nn.train_tuned import LLMTrainerTuned


def parse_args():
    p = argparse.ArgumentParser("Train LLM with tuned optimizer/LoRA settings")
    # Data
    p.add_argument('--json_file', type=str, required=True)
    p.add_argument('--features_file', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='logs')
    p.add_argument('--exp_name', type=str, default='interpert_tuned')
    p.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    # Core model args (delegated to imported functions via env/config)
    p.add_argument('--max_seq_length', type=int, default=128)
    # Training
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--warmup_epochs', type=int, default=2)
    p.add_argument('--early_stopping', type=int, default=10)
    p.add_argument('--max_iter', type=int, default=-1, help='-1 for full epoch')
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--gradient_checkpointing', action='store_true', default=True)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--random_seed', type=int, default=42)
    # Resume
    p.add_argument('--resume_path', type=str, default=None)
    p.add_argument('--checkpoint_dir', type=str, default=None)
    # Tunables
    p.add_argument('--lora_lr_scale', type=float, default=0.5, help='relative LR for LoRA params')
    p.add_argument('--label_smoothing', type=float, default=0.1)
    return p.parse_args()


def build_param_groups(model, base_lr, weight_decay, lora_lr_scale=0.5):
    """Create optimizer param groups:
    - No weight decay for biases and norm weights
    - Separate LoRA params group with scaled LR
    """
    norm_keywords = ['norm', 'ln', 'layer_norm', 'rmsnorm', 'attention_norm', 'ffn_norm']
    lora_params, wd0_params, wd_params = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'lora' in name:
            lora_params.append(p)
            continue
        if name.endswith('.bias') or any(k in name.lower() for k in norm_keywords):
            wd0_params.append(p)
        else:
            wd_params.append(p)

    groups = []
    if wd_params:
        groups.append({'params': wd_params, 'lr': base_lr, 'weight_decay': weight_decay})
    if wd0_params:
        groups.append({'params': wd0_params, 'lr': base_lr, 'weight_decay': 0.0})
    if lora_params:
        groups.append({'params': lora_params, 'lr': base_lr * lora_lr_scale, 'weight_decay': 0.0})
    return groups


def create_optimizer_and_scheduler_tuned(model, args, train_loader):
    # Freeze logic is handled in create_model_memory_optimized; ensure only trainables pass
    groups = build_param_groups(model.module if isinstance(model, DDP) else model,
                                args.learning_rate, args.weight_decay, args.lora_lr_scale)
    optimizer = torch.optim.AdamW(groups)

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


def main():
    args = parse_args()
    print(args)
    date = __import__('datetime').datetime.now().strftime('%Y-%m-%d-%H-%M')
    args.output_dir = os.path.join(args.output_dir, date)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup
    local_rank, world_size, _ = setup()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # Data
    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, local_rank)
    tokenizer = train_loader.dataset.tokenizer

    # Model
    model = create_model_memory_optimized(args, local_rank)

    # Optimizer
    optimizer, scheduler, scaler = create_optimizer_and_scheduler_tuned(model, args, train_loader)

    # Trainer
    trainer = LLMTrainerTuned(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=local_rank,
        world_size=world_size,
        output_dim=1,
        scheduler=None,
        max_iter=args.max_iter,
        log_path=args.output_dir,
        exp_name=args.exp_name,
        lora_params=json.load(open(os.path.join(ROOT_DIR, 'src', 'llm_config.json')))['lora_params'],
        scaler=scaler,
        use_amp=args.use_amp,
        max_grad_norm=args.max_grad_norm,
        tokenizer=tokenizer,
        label_smoothing=args.label_smoothing,
    )
    trainer.scheduler = scheduler

    if local_rank == 0:
        save_config(args, args.output_dir)

    # Train
    if True:
        print("Starting tuned training...")
        trainer.evaluate_validation_samples(local_rank, 0)
        _ = trainer.fit(
            num_epochs=args.num_epochs,
            device=local_rank,
            early_stopping=args.early_stopping,
            best='loss',
        )


if __name__ == '__main__':
    print('='*80)
    print('LLM TRAINING (TUNED)')
    print('='*80)
    main()

