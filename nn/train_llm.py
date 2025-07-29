import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, Any, List

# Import our custom modules
from llama3.llama.model_single_gpu import Transformer, ModelArgs  # Your non-parallel LLaMA
from llama3.llama.tokenizer import Tokenizer
from data.star_dataset import StarDataset, setup_dataloaders, create_sample_data
from nn.llm import MultimodalLlamaModel, create_multimodal_llama, train_step, evaluate_model, MODEL_PATH, TOKENIZER_PATH
from util.utils import merge_dataframes_and_filter_array


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer"""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x shape: (..., in_features)
        # LoRA forward: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return self.dropout(lora_out)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""

    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank, alpha, dropout
        )

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_out = self.original_layer(x)
        lora_out = self.lora(x)
        return original_out + lora_out


def apply_lora_to_model(model: nn.Module, target_modules: List[str], rank: int = 16, alpha: float = 16.0,
                        dropout: float = 0.1):
    """Apply LoRA to specified modules in the model"""
    lora_modules = {}

    def replace_with_lora(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this is a target module
            if any(target in full_name for target in target_modules) and isinstance(child_module, nn.Linear):
                print(f"Applying LoRA to: {full_name}")
                lora_layer = LoRALinear(child_module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)
                lora_modules[full_name] = lora_layer
            else:
                # Recursively apply to children
                replace_with_lora(child_module, full_name)

    replace_with_lora(model)
    return lora_modules


class StarLlamaTrainer:
    """Complete training pipeline for Star LLaMA with multimodal capabilities and LoRA"""

    def __init__(self,
                 model_checkpoint_path: str,
                 multimodal_checkpoints: str,
                 tokenizer_path: str,
                 latent_features: np.ndarray,
                 metadata_df: pd.DataFrame,
                 config: Dict[str, Any] = None):
        """
        Initialize the trainer

        Args:
            model_checkpoint_path: Path to LLaMA checkpoint directory
            tokenizer_path: Path to tokenizer.model file
            latent_features: Array of shape (n_samples, latent_dim)
            metadata_df: DataFrame with star metadata
            config: Training configuration dictionary
        """
        self.model_checkpoint_path = model_checkpoint_path
        self.multimodal_checkpoints = multimodal_checkpoints
        self.tokenizer_path = tokenizer_path
        self.latent_features = latent_features
        self.metadata_df = metadata_df
        self.num_samples = len(latent_features)

        # Default configuration
        self.config = {
            'batch_size': 4,
            'learning_rate': 5e-4,  # Higher LR for LoRA
            'num_epochs': 10,
            'max_sequence_length': 512,
            'noise_std': 0.01,
            'train_split': 0.8,
            'save_every_n_epochs': 2,
            'eval_every_n_steps': 50,
            'special_token': '<STAR_DATA>',
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,

            # Freezing configuration
            'freeze_strategy': 'encoder_only',  # 'encoder_only', 'lora', 'none'
            'encoder_only_epochs': 3,  # Train only encoder for first N epochs
            'encoder_lr_multiplier': 1.0,  # LR multiplier for encoder vs base model
            'lora_lr_multiplier': 1.0,  # LR multiplier for LoRA layers

            # LoRA configuration
            'lora_rank': 16,
            'lora_alpha': 16.0,
            'lora_dropout': 0.1,
            'lora_target_modules': [
                'layers.*.attention.wq',
                'layers.*.attention.wk',
                'layers.*.attention.wv',
                'layers.*.attention.wo',
                'layers.*.feed_forward.w1',
                'layers.*.feed_forward.w2',
                'layers.*.feed_forward.w3'
            ],
        }

        if config:
            self.config.update(config)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.current_freeze_state = None
        self.lora_modules = {}

    def setup(self):
        """Setup all components for training"""
        print("Setting up training components...")

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = Tokenizer(model_path=self.tokenizer_path)

        # Load base LLaMA model
        print("Loading base model...")
        self.base_model = self._load_base_model()

        # Create multimodal model
        print("Creating multimodal model...")
        latent_dim = self.latent_features.shape[1]
        self.model = create_multimodal_llama(
            self.base_model,
            latent_dim,
            self.tokenizer,
            self.config['special_token']
        )

        if self.multimodal_checkpoints is not None:
            print(f"Loading multimodal checkpoints from {self.multimodal_checkpoints}...")
            self.model.load_state_dict(torch.load(self.multimodal_checkpoints)['model_state_dict'])

        self.model.to(self.device)

        # Setup data loaders
        print("Setting up data loaders...")
        self.train_loader, self.val_loader = setup_dataloaders(
            self.latent_features,
            self.metadata_df,
            self.tokenizer,
            batch_size=self.config['batch_size'],
            train_split=self.config['train_split'],
            noise_std=self.config['noise_std']
        )

        # Setup optimizer and scheduler
        print("Setting up optimizer...")
        self._setup_optimizer()

        print("Setup complete!")

    def _load_base_model(self) -> Transformer:
        """Load the base LLaMA model"""
        # Load model parameters
        with open(Path(self.model_checkpoint_path) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args = ModelArgs(
            max_batch_size=self.config['batch_size'] * 2,
            max_seq_len=self.config['max_sequence_length'],
            **params,
        )

        # Create model
        model = Transformer(model_args)

        # Load checkpoint
        checkpoints = sorted(Path(self.model_checkpoint_path).glob("*.pth"))
        if checkpoints:
            print(f"Loading checkpoint: {checkpoints[0]}")
            checkpoint = torch.load(checkpoints[0], map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
        else:
            print("Warning: No checkpoint found, using random initialization")

        return model

    def _apply_lora(self):
        """Apply LoRA to the model"""
        print("Applying LoRA layers...")

        # Convert wildcard patterns to actual module names
        target_modules = []
        for pattern in self.config['lora_target_modules']:
            if '*' in pattern:
                # Find matching modules
                for name, _ in self.model.named_modules():
                    import re
                    pattern_regex = pattern.replace('*', r'[^.]*')
                    if re.match(f"^{pattern_regex}$", name):
                        target_modules.append(name)
            else:
                target_modules.append(pattern)

        print(f"Target modules for LoRA: {target_modules[:5]}...")  # Show first 5

        self.lora_modules = apply_lora_to_model(
            self.model,
            target_modules,
            rank=self.config['lora_rank'],
            alpha=self.config['lora_alpha'],
            dropout=self.config['lora_dropout']
        )

        print(f"Applied LoRA to {len(self.lora_modules)} modules")

    def _apply_freeze_strategy(self, strategy: str):
        """Apply different freezing strategies to the model"""
        print(f"Applying freeze strategy: {strategy}")

        if strategy == 'encoder_only':
            # Freeze everything except latent encoder
            for name, param in self.model.named_parameters():
                if 'latent_encoder' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif strategy == 'lora':
            # Apply LoRA if not already applied
            if not self.lora_modules:
                self._apply_lora()

            # Freeze base model, enable encoder and LoRA parameters
            for name, param in self.model.named_parameters():
                if 'latent_encoder' in name:
                    param.requires_grad = True
                elif 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif strategy == 'none':
            # Unfreeze everything
            for param in self.model.parameters():
                param.requires_grad = True

        self.current_freeze_state = strategy

        # Print what's trainable
        trainable_modules = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                module_type = 'lora' if 'lora' in name else name.split('.')[0]
                trainable_modules.add(module_type)

        print(f"Trainable module types: {list(trainable_modules)}")

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Apply freeze strategy
        self._apply_freeze_strategy(self.config['freeze_strategy'])

        # Get parameter groups with different learning rates
        encoder_params = []
        lora_params = []
        base_params = []

        for name, param in self.model.named_parameters():
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
                'lr': self.config['learning_rate'] * self.config['encoder_lr_multiplier'],
                'name': 'encoder'
            })

        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': self.config['learning_rate'] * self.config['lora_lr_multiplier'],
                'name': 'lora'
            })

        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': self.config['learning_rate'] * 0.1,  # Lower LR for base model
                'name': 'base_model'
            })

        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config['weight_decay']
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.config['num_epochs']
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config['learning_rate'] * 0.01
        )

        # Print parameter info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")

        # Debug: Print specific trainable parameters by type
        param_counts = {'encoder': 0, 'lora': 0, 'base': 0}
        for name, param in self.model.named_parameters():
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

    def debug_model_setup(self):
        """Debug method to check model setup"""
        print("\n=== DEBUG: Model Setup ===")

        # Check if model has latent_encoder
        has_encoder = hasattr(self.model, 'latent_encoder')
        print(f"Model has latent_encoder: {has_encoder}")

        if has_encoder:
            encoder_params = sum(p.numel() for p in self.model.latent_encoder.parameters())
            print(f"Latent encoder parameters: {encoder_params:,}")

        # Check LoRA modules
        if self.lora_modules:
            lora_params = sum(sum(p.numel() for p in module.lora.parameters())
                              for module in self.lora_modules.values())
            print(f"LoRA parameters: {lora_params:,}")

        # Check parameter requires_grad status by type
        param_types = {'encoder': 0, 'lora': 0, 'base': 0, 'frozen': 0}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                param_types['frozen'] += param.numel()
            elif 'latent_encoder' in name:
                param_types['encoder'] += param.numel()
            elif 'lora' in name:
                param_types['lora'] += param.numel()
            else:
                param_types['base'] += param.numel()

        print("Parameter status by type:")
        for param_type, count in param_types.items():
            print(f"  {param_type}: {count:,} parameters")

        # Test forward pass
        print("Testing forward pass...")
        try:
            # Create dummy batch
            dummy_input = torch.randint(0, 1000, (1, 10)).to(self.device)
            dummy_latent = torch.randn(1, self.latent_features.shape[1]).to(self.device)
            dummy_pos = torch.tensor([5]).to(self.device)

            with torch.no_grad():
                output = self.model(dummy_input, dummy_latent, dummy_pos, start_pos=0)
                print(f"Forward pass successful! Output shape: {output.shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")

        print("=== End Debug ===\n")

    def update_freeze_strategy(self, new_strategy: str):
        """Update freezing strategy and reinitialize optimizer"""
        print(f"Updating freeze strategy from '{self.current_freeze_state}' to '{new_strategy}'")

        # Update config and reapply
        self.config['freeze_strategy'] = new_strategy
        self._setup_optimizer()

    def train(self):
        """Main training loop"""
        print("Starting training...")

        best_val_loss = float('inf')
        step = 0

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")

            # Handle automatic freeze strategy changes
            if (self.config['freeze_strategy'] == 'encoder_only' and
                    epoch >= self.config['encoder_only_epochs'] and
                    self.current_freeze_state == 'encoder_only'):
                print(f"Switching from encoder-only to LoRA training at epoch {epoch + 1}")
                self.update_freeze_strategy('lora')

            # Training phase
            self.model.train()
            epoch_train_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                loss = train_step(self.model, batch, self.optimizer, self.device)
                epoch_train_loss += loss
                num_batches += 1
                step += 1

                # Update learning rate
                self.scheduler.step()

                # Log progress
                if batch_idx % 10 == 0:
                    current_lrs = [group['lr'] for group in self.optimizer.param_groups]
                    lr_str = ', '.join([f"{lr:.2e}" for lr in current_lrs])
                    print(f"  Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {loss:.4f}, LRs: [{lr_str}], "
                          f"Freeze: {self.current_freeze_state}")

                # Evaluation
                if step % self.config['eval_every_n_steps'] == 0:
                    val_loss = evaluate_model(self.model, self.val_loader, self.device)
                    print(f"  Step {step} - Validation Loss: {val_loss:.4f}")

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_checkpoint(epoch, step, val_loss, is_best=True)

                    self.model.train()  # Back to training mode

            # Epoch summary
            avg_train_loss = epoch_train_loss / num_batches
            val_loss = evaluate_model(self.model, self.val_loader, self.device)

            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Average Train Loss: {avg_train_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Current Freeze Strategy: {self.current_freeze_state}")

            # Save checkpoint
            if (epoch + 1) % self.config['save_every_n_epochs'] == 0:
                self._save_checkpoint(epoch, step, val_loss)

            # Generate sample response
            self._generate_sample_response(epoch)

    def _save_checkpoint(self, epoch: int, step: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'lora_modules': list(self.lora_modules.keys()) if self.lora_modules else []
        }

        if is_best:
            filename = f'checkpoints/best_model_lora_{self.num_samples}_samples.pt'
        else:
            filename = f'checkpoints/checkpoint_lora_epoch_{epoch + 1}_{self.num_samples}_samples.pt'

        torch.save(checkpoint, filename)
        print(f"Saved checkpoint: {filename}")

    def _generate_sample_response(self, epoch: int):
        """Generate a sample response to see how the model is doing"""
        try:
            # Get a random sample from validation set
            sample_idx = np.random.randint(0, len(self.val_loader.dataset))
            sample = self.val_loader.dataset[sample_idx]

            # Create a question
            question = f"Describe the physical parameters of this star {self.config['special_token']}"
            question_tokens = self.tokenizer.encode(question, bos=True, eos=False)

            # Generate response
            response = self.model.generate_response(
                question_tokens,
                sample['latent_features'],
                self.tokenizer,
                max_new_tokens=50,
                temperature=0.7,
                device=self.device
            )

            print(f"\n--- Sample Generation (Epoch {epoch + 1}) ---")
            print(f"Question: {question}")
            print(f"Generated Response: {response}")
            print(f"Expected Answer: {sample['answer_text']}")
            print("--- End Sample ---\n")

        except Exception as e:
            print(f"Error generating sample: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint

    def evaluate_on_samples(self, num_samples: int = 10):
        """Evaluate model on a few samples and print results"""
        print(f"\n--- Evaluating on {num_samples} samples ---")

        self.model.eval()
        questions = [
            f"Describe the physical parameters of this star {self.config['special_token']}",
            f"What are the characteristics of this stellar object {self.config['special_token']}?",
            f"What type of star is this {self.config['special_token']}?",
        ]

        with torch.no_grad():
            for i in range(min(num_samples, len(self.val_loader.dataset))):
                sample = self.val_loader.dataset[i]
                question = str(np.random.choice(questions))
                question_tokens = self.tokenizer.encode(question, bos=True, eos=False)

                response = self.model.generate_response(
                    question_tokens,
                    sample['latent_features'],
                    self.tokenizer,
                    max_new_tokens=50,
                    temperature=0.7,
                    device=self.device
                )

                print(f"\nSample {i + 1}:")
                print(f"Question: {question}")
                print(f"Generated: {response}")
                print(f"Expected: {sample['answer_text']}")

        print("--- End Evaluation ---\n")

    def print_model_status(self):
        """Print current model parameter status"""
        print("\n=== Model Parameter Status ===")

        total_params = 0
        param_types = {'encoder': 0, 'lora': 0, 'base_trainable': 0, 'base_frozen': 0}

        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count

            if 'latent_encoder' in name:
                if param.requires_grad:
                    param_types['encoder'] += param_count
            elif 'lora' in name:
                if param.requires_grad:
                    param_types['lora'] += param_count
            else:
                if param.requires_grad:
                    param_types['base_trainable'] += param_count
                else:
                    param_types['base_frozen'] += param_count

        trainable_params = param_types['encoder'] + param_types['lora'] + param_types['base_trainable']

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
        print(f"Current freeze strategy: {self.current_freeze_state}")

        print("\nParameter breakdown:")
        for param_type, count in param_types.items():
            if count > 0:
                pct = 100 * count / total_params
                status = "✓" if 'trainable' in param_type or param_type in ['encoder', 'lora'] else "✗"
                print(f"  {status} {param_type}: {count:,} ({pct:.1f}%)")

        if self.lora_modules:
            print(f"\nLoRA modules applied: {len(self.lora_modules)}")
            print(f"LoRA rank: {self.config['lora_rank']}, alpha: {self.config['lora_alpha']}")

        print("=" * 40)


def main(num_samples=5000):
    """Main function to run training"""

    # Configuration for progressive training with LoRA
    config = {
        'batch_size': 4,
        'learning_rate': 5e-4,  # Higher LR for LoRA
        'num_epochs': 8,
        'max_sequence_length': 256,
        'noise_std': 0.02,
        'train_split': 0.8,
        'eval_every_n_steps': 25,
        'save_every_n_epochs': 2,

        # Progressive training: start with encoder only, then LoRA
        'freeze_strategy': 'encoder_only',
        'encoder_only_epochs': 3,  # Switch to LoRA after 3 epochs
        'encoder_lr_multiplier': 1.0,
        'lora_lr_multiplier': 1.0,

        # LoRA configuration
        'lora_rank': 16,
        'lora_alpha': 16.0,
        'lora_dropout': 0.1,
        'lora_target_modules': [
            'layers.*.attention.wq',
            'layers.*.attention.wk',
            'layers.*.attention.wv',
            'layers.*.attention.wo',
            'layers.*.feed_forward.w1',
            'layers.*.feed_forward.w2',
            'layers.*.feed_forward.w3'
        ],
    }

    # Paths (update these to your actual paths)
    model_checkpoint_path = MODEL_PATH
    tokenizer_path = TOKENIZER_PATH
    multimmodal_checkpoints = 'checkpoints/best_model.pt'

    # Create sample data for testing
    print("Creating sample data...")
    latent_features = np.load('logs/2025-07-29/features.npy')
    metadata_df = pd.read_csv('logs/2025-07-29/info.csv')
    metadata_df = metadata_df.loc[:, ~metadata_df.columns.str.contains('^Unnamed')]

    berger_lamost = pd.read_csv('tables/lamost_dr8_with_berger_labels.csv')
    berger_lamost = berger_lamost.loc[:, ~berger_lamost.columns.str.contains('^Unnamed')]

    metadata_df, latent_features = merge_dataframes_and_filter_array(
        berger_lamost, metadata_df, 'obsid', 'obsid', latent_features
    )

    latent_features = latent_features[:num_samples, :]
    metadata_df = metadata_df.iloc[:num_samples]

    # Create trainer
    trainer = StarLlamaTrainer(
        model_checkpoint_path=model_checkpoint_path,
        multimodal_checkpoints=None,
        tokenizer_path=tokenizer_path,
        latent_features=latent_features,
        metadata_df=metadata_df,
        config=config
    )

    try:
        trainer.setup()

        # Debug model setup
        trainer.debug_model_setup()

        # Debug special token detection
        print("Debugging special token detection...")
        trainer.train_loader.dataset.debug_special_token(0)

        # Test first batch to catch gradient issues early
        print("Testing first batch...")
        first_batch = next(iter(trainer.train_loader))

        # Check special token positions
        special_positions = first_batch['special_token_positions']
        print(f"Special token positions: {special_positions}")

        # Quick gradient test
        trainer.model.train()
        input_ids = first_batch['input_ids'].to(trainer.device)
        latent_features = first_batch['latent_features'].to(trainer.device)
        special_token_positions = first_batch['special_token_positions'].to(trainer.device)

        logits = trainer.model(input_ids, latent_features, special_token_positions, start_pos=0)
        print(f"Test forward pass successful! Logits requires_grad: {logits.requires_grad}")

        # Print initial model status
        trainer.print_model_status()

        # Train with automatic freeze strategy transitions
        trainer.train()

        # Final evaluation
        trainer.evaluate_on_samples(num_samples=5)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

    print("Training completed!")


if __name__ == "__main__":
    main()