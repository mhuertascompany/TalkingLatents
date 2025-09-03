import torch
import torch.optim as optim
import torch.nn as nn
import json
from pathlib import Path
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

os.system('pip install tiktoken fairscale fire blobfile')

from data.dataset_llm import StellarDatasetManager
from nn.state_space_llm import PhysicsInformedEvolutionaryLlama, apply_lora_to_model
from llama3.llama.model import Transformer, ModelArgs
from llama3.llama.tokenizer import Tokenizer
from nn.train import *

TOKENIZER_PATH = "/data/.llama/Llama3.1-8B/tokenizer.model"
MODEL_PATH = "/data/.llama/Llama3.1-8B"
custom_special_tokens = [
        "STAR_DATA_0", "STAR_DATA_1", "STAR_DATA_2", "STAR_DATA_3", "STAR_DATA_4",
        "STAR_DATA_5", "STAR_DATA_6", "STAR_DATA_7", "STAR_DATA_8", "STAR_DATA_9"
    ]


def _load_base_model() -> Transformer:
    """Load the base LLaMA model"""
    with open(Path(MODEL_PATH) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(
        max_batch_size=4,
        max_seq_len=1024,
        **params,
    )

    model = Transformer(model_args)

    checkpoints = sorted(Path(MODEL_PATH).glob("*.pth"))
    if checkpoints:
        print(f"Loading checkpoint: {checkpoints[0]}")
        checkpoint = torch.load(checkpoints[0], map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("Warning: No checkpoint found, using random initialization")

    return model

def _apply_lora(model, config):
        """Apply LoRA to the model"""
        print("Applying LoRA layers...")

        # First, let's see what modules actually exist in the model
        print("Available modules in model:")
        all_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                all_modules.append(name)

        print(f"Found {len(all_modules)} Linear modules:")
        for i, name in enumerate(all_modules[:10]):  # Show first 10
            print(f"  {name}")
        if len(all_modules) > 10:
            print(f"  ... and {len(all_modules) - 10} more")

        # Convert wildcard patterns to actual module names
        target_modules = []
        import re

        for pattern in config['lora_target_modules']:
            if '*' in pattern:
                # Convert pattern to regex, handling multiple wildcards
                pattern_regex = pattern.replace('.', r'\.').replace('*', r'[^.]+')
                pattern_regex = f"^{pattern_regex}$"

                # Find matching modules
                matched_modules = []
                for name in all_modules:
                    if re.match(pattern_regex, name):
                        matched_modules.append(name)
                        target_modules.append(name)

                print(f"Pattern '{pattern}' matched {len(matched_modules)} modules")
                if matched_modules:
                    print(f"  Examples: {matched_modules[:3]}")
            else:
                if pattern in all_modules:
                    target_modules.append(pattern)
                    print(f"Direct match: {pattern}")
                else:
                    print(f"Warning: Pattern '{pattern}' not found in model")

        print(f"\nTotal target modules for LoRA: {len(target_modules)}")
        if target_modules:
            print("Target modules:")
            for i, name in enumerate(target_modules[:5]):
                print(f"  {name}")
            if len(target_modules) > 5:
                print(f"  ... and {len(target_modules) - 5} more")

        if not target_modules:
            print("ERROR: No target modules found! Check your lora_target_modules patterns.")
            print("Available Linear module examples:")
            for name in all_modules[:10]:
                print(f"  {name}")
            return

        lora_modules = apply_lora_to_model(
            model,
            target_modules,
            rank=config['lora_rank'],
            alpha=config['lora_alpha'],
            dropout=config['lora_dropout']
        )

        print(f"Successfully applied LoRA to {len(lora_modules)} modules")

        return lora_modules

def _setup_optimizer(model, config, train_loader):
        """Setup optimizer and learning rate scheduler"""
        # Apply freeze strategy
        # _apply_freeze_strategy(config['freeze_strategy'])

        # Get parameter groups with different learning rates
        encoder_params = []
        lora_params = []
        base_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if 'latent_encoder' in name:
                encoder_params.append(param)
            elif 'lora' in name:
                lora_params.append(param)
            else:
                base_params.append(param)

        # Check if we have any trainable parameters
        if not encoder_params and not lora_params and not base_params:
            raise ValueError("No trainable parameters found! Check your freeze strategy.")

        param_groups = []
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': config['learning_rate'] * config['encoder_lr_multiplier'],
                'name': 'encoder'
            })

        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': config['learning_rate'] * config['lora_lr_multiplier'],
                'name': 'lora'
            })

        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': config['learning_rate'] * 0.1,  # Lower LR for base model
                'name': 'base_model'
            })

        optimizer = optim.AdamW(
            param_groups,
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * config['num_epochs']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=config['learning_rate'] * 0.01
        )

        # Print parameter info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")

        # Debug: Print specific trainable parameters by type
        param_counts = {'encoder': 0, 'lora': 0, 'base': 0}
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'latent_encoder' in name:
                    param_counts['encoder'] += param.numel()
                elif 'lora' in name:
                    param_counts['lora'] += param.numel()
                else:
                    param_counts['base'] += param.numel()

        print("Trainable parameter breakdown:")
        for param_type, count in param_counts.items():
            if count > 0:
                print(f"  {param_type}: {count:,} parameters")

        return optimizer, scheduler

def initialize_llm(device, config):
    tokenizer = Tokenizer(
        model_path=TOKENIZER_PATH,
        custom_special_tokens=custom_special_tokens
    )
    base_model = _load_base_model()
    base_model = base_model.to(device)


    # The special tokens are already in the tokenizer, so we just need to tell the model about them
    special_tokens_dict = tokenizer.custom_token_ids

    model = PhysicsInformedEvolutionaryLlama(
        base_model=base_model,
        latent_dim=config['features_dim'],
        special_tokens=special_tokens_dict,
        max_stages=config['max_stages'],
        evolution_model_type="lstm",
    ).to(device)

    manager = StellarDatasetManager(config['json_file_path'], device)
    manager.load_data()

    # Create splits - tokenizer already has special tokens configured
    _ = manager.create_splits(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        tokenizer=tokenizer,
        max_length=512
    )

    optimizer, scheduler = _setup_optimizer(model, config, manager.get_dataloaders(batch_size=config['batch_size'], num_workers=0)['train'])
    return model, manager, optimizer, scheduler, tokenizer



def main():
    with open("/data/TalkingLatents/src/llm_config.json", "r") as f:
        config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_manager, optimizer, scheduler = initialize_llm(device, config)
    b_size = config['batch_size']
    train_loader = data_manager.get_dataloaders(batch_size=b_size, num_workers=0)['train']
    val_loader = data_manager.get_dataloaders(batch_size=b_size, num_workers=0)['val']
    test_loader = data_manager.get_dataloaders(batch_size=b_size, num_workers=0)['test']

    trainer = LLMTrainer(model=model,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         criterion=None,
                         train_dataloader=train_loader,
                         val_dataloader=val_loader,
                         device=device,
                         lora_params=config['lora_params'],)

    trainer.fit(num_epochs=config['num_epochs'], device=device)


if __name__ == "__main__":
    main()