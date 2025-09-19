import os
import sys
import json
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.simple_questions import (
    setup,
    parse_args as base_parse_args,
    create_datasets_and_loaders,
    create_optimizer_and_scheduler,
    save_config,
    _load_llm_model,
    _load_spectra_model,
    print_detailed_memory,
)
from nn.llm_multi import MultimodalLlamaModelMultiTokens
from nn.train import LLMTrainer


def build_model_multitok(args, device):
    print("Loading LLM model...")
    llm = _load_llm_model(args)
    if args.llm_precision == 'fp16':
        llm.half(); print("✓ LLM weights cast to float16")
    elif args.llm_precision == 'bf16' and torch.cuda.is_bf16_supported():
        llm.to(dtype=torch.bfloat16); print("✓ LLM weights cast to bfloat16")
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
    ).to(device)
    return model


def main():
    args = base_parse_args()
    # allow override from env for quick tests
    date = __import__('datetime').datetime.now().strftime('%Y-%m-%d-%H-%M')
    args.output_dir = os.path.join(args.output_dir, date)
    os.makedirs(args.output_dir, exist_ok=True)

    local_rank, world_size, _ = setup()
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    print("Creating datasets and dataloaders...")
    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, local_rank)
    tokenizer = train_loader.dataset.tokenizer

    print("Creating multitoken multimodal model...")
    model = build_model_multitok(args, local_rank)
    print_detailed_memory()

    if world_size > 1:
        print(f"Wrapping with DDP (world_size={world_size})")
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        print("Single GPU - no DDP")

    print("Creating optimizer and scheduler...")
    optimizer, scheduler, scaler = create_optimizer_and_scheduler(model, args, train_loader)

    print("Creating trainer (tuned LoRA config)...")
    # Load tuned LoRA config (attention-only by default)
    tuned_cfg_path = os.path.join(ROOT_DIR, 'src', 'llm_config_tuned.json')
    if os.path.isfile(tuned_cfg_path):
        with open(tuned_cfg_path, 'r') as f:
            tuned_cfg = json.load(f)
        lora_params = tuned_cfg.get('lora_params', {})
    else:
        # Fallback to base config if tuned not found
        base_cfg_path = os.path.join(ROOT_DIR, 'src', 'llm_config.json')
        with open(base_cfg_path, 'r') as f:
            base_cfg = json.load(f)
        lora_params = base_cfg.get('lora_params', {})

    trainer = LLMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=local_rank,
        world_size=world_size,
        output_dim=1,
        scheduler=None,
        max_iter=args.max_iter,
        log_path=args.output_dir,
        exp_name=args.exp_name,
        lora_params=lora_params,
        scaler=scaler,
        use_amp=args.use_amp,
        max_grad_norm=args.max_grad_norm,
    )
    trainer.scheduler = scheduler
    trainer.tokenizer = tokenizer

    if local_rank == 0:
        save_config(args, args.output_dir)

    if args.train:
        trainer.evaluate_validation_samples(local_rank, 0)
        _ = trainer.fit(
            num_epochs=args.num_epochs,
            device=local_rank,
            early_stopping=args.early_stopping,
            best='loss'
        )


if __name__ == '__main__':
    print("="*80)
    print("CLIP MULTIMODAL STELLAR MODEL TRAINING (Multi spectral tokens)")
    print("="*80)
    main()
