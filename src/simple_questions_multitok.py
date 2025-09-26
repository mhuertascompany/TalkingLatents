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


class DummyTopologyModel(torch.nn.Module):
    """Small stand-in model for local smoke tests."""

    def __init__(self, vocab_size: int = 32000, embed_dim: int = 256,
                 latent_dim: int = 512, physics_dim: int = 0):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.position = torch.nn.Parameter(torch.zeros(512, embed_dim))
        torch.nn.init.normal_(self.position, mean=0.0, std=0.02)
        self.token_ff = torch.nn.Linear(embed_dim, embed_dim)
        self.output_proj = torch.nn.Linear(embed_dim, vocab_size)
        self.text_to_latent = torch.nn.Linear(embed_dim, latent_dim)
        self.physics_dim = physics_dim
        if physics_dim > 0:
            self.latent_to_physics = torch.nn.Linear(latent_dim, physics_dim)
        else:
            self.latent_to_physics = None
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, input_spectra: torch.Tensor,
                special_token_positions: torch.Tensor, start_pos: int = 0,
                question_start_indices=None, answer_start_indices=None,
                neighbor_latents=None, neighbor_mask=None):
        batch, seq = input_ids.shape
        device = input_ids.device
        emb = self.embedding(input_ids)
        pos = self.position[:seq].unsqueeze(0)
        emb = emb + pos
        hidden = torch.tanh(self.token_ff(emb))
        logits = self.output_proj(hidden)

        spectra = input_spectra.reshape(batch, -1).float()
        latent_target = spectra
        latent_recon = latent_target

        pooled = hidden.mean(dim=1)
        pred_latent = self.text_to_latent(pooled)

        physics_pred = None
        if self.latent_to_physics is not None:
            physics_pred = self.latent_to_physics(pred_latent)

        neighbor_logits = None
        if neighbor_latents is not None and neighbor_latents.numel() > 0:
            neighbor_vec = neighbor_latents.reshape(batch, neighbor_latents.size(1), -1).to(pred_latent.dtype)
            neighbor_logits = torch.matmul(neighbor_vec, pred_latent.unsqueeze(-1)).squeeze(-1)

        return {
            "logits": logits,
            "h": hidden,
            "latent_recon_from_tokens": latent_recon,
            "latent_target": latent_target,
            "pred_latent_from_text": pred_latent,
            "physics_pred": physics_pred,
            "neighbor_logits": neighbor_logits,
            "neighbor_mask": neighbor_mask,
        }

    @torch.no_grad()
    def generate_response_from_batch(self, batch_data: dict, batch_idx: int = 0,
                                     tokenizer=None, **kwargs):
        input_text = batch_data.get('input_texts', [''])[batch_idx]
        target_text = batch_data.get('target_texts', [''])[batch_idx]
        return 'dummy-response', input_text, target_text, []


def build_model_multitok(args, device):
    if getattr(args, 'use_dummy_llm', False):
        print("Using dummy topology model for local testing")
        physics_dim = len(getattr(args, 'physics_keys', []) or [])
        dummy = DummyTopologyModel(
            vocab_size=32000,
            embed_dim=getattr(args, 'hidden_dim', 256),
            latent_dim=args.spectral_embedding_dim,
            physics_dim=physics_dim,
        ).to(device)
        return dummy

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
        physics_dim=len(getattr(args, 'physics_keys', []) or []),
    ).to(device)
    # Keep projector in float32 for numeric stability with GradScaler
    # (base model runs in fp16/bf16; features are cast to projector dtype inside model)
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

    if torch.cuda.is_available():
        runtime_device = torch.device('cuda', local_rank)
    else:
        runtime_device = torch.device('cpu')

    print("Creating datasets and dataloaders...")
    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, local_rank)
    tokenizer = train_loader.dataset.tokenizer

    print("Creating multitoken multimodal model...")
    model = build_model_multitok(args, runtime_device)

    # Freeze large submodules BEFORE wrapping with DDP so the reducer
    # only tracks truly trainable parameters (avoids unused-grad errors).
    base = model
    if isinstance(base, DDP):
        base = base.module
    if hasattr(base, 'base_model') and base.base_model is not None:
        for p in base.base_model.parameters():
            p.requires_grad = False
        print("✓ Frozen base LLaMA parameters")
    if hasattr(base, 'fm_model') and base.fm_model is not None:
        for p in base.fm_model.parameters():
            p.requires_grad = False
        print("✓ Frozen spectral FM parameters")
    print_detailed_memory()

    if world_size > 1 and torch.cuda.is_available():
        print(f"Wrapping with DDP (world_size={world_size})")
        # All large modules are frozen; but LoRA may be applied later and
        # not participate immediately. Use find_unused_parameters=True to
        # avoid reducer errors in early epochs.
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            broadcast_buffers=False,
            bucket_cap_mb=25,
            gradient_as_bucket_view=True,
        )
    else:
        print("Single process - no DDP")

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
        device=runtime_device,
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
        # Optional auxiliary invertibility loss weight (default 0 if arg missing)
        lambda_feat=getattr(args, 'lambda_feat', 0.0),
        lambda_text=getattr(args, 'lambda_text', 0.0),
        lambda_retrieval=getattr(args, 'lambda_retrieval', 0.0),
        lambda_physics=getattr(args, 'lambda_physics', 0.0),
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
