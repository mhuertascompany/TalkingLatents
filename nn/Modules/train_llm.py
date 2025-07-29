import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import numpy as np


class StarDataset(Dataset):
    """
    Dataset for star latent vectors and their corresponding parameter descriptions
    """

    def __init__(
            self,
            latent_vectors: np.ndarray,
            descriptions: List[str],
            tokenizer,
            max_length: int = 512
    ):
        self.latent_vectors = torch.FloatTensor(latent_vectors)
        self.descriptions = descriptions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.latent_vectors)

    def __getitem__(self, idx):
        latent_vector = self.latent_vectors[idx]
        description = self.descriptions[idx]

        # Create input prompt
        prompt = "This star has the following parameters:"
        full_text = f"{prompt} {description}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'latent_vector': latent_vector,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'prompt': prompt,
            'description': description
        }


class TrainingPipeline:
    """
    Two-stage training pipeline for multimodal LLM
    """

    def __init__(
            self,
            model: nn.Module,
            train_dataset: Dataset,
            val_dataset: Dataset,
            config: Dict
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 2)
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 2)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Initialize logging
        if config.get('use_wandb', False):
            wandb.init(project=config['project_name'], config=config)

    def create_optimizer_and_scheduler(self, stage: str):
        """Create optimizer and scheduler for specific training stage"""
        if stage == "encoder_only":
            # Only optimize latent encoder parameters
            parameters = self.model.latent_encoder.parameters()
            lr = self.config['encoder_lr']
        elif stage == "full_model":
            # Optimize all parameters
            parameters = self.model.parameters()
            lr = self.config['full_model_lr']
        else:
            raise ValueError(f"Unknown stage: {stage}")

        optimizer = AdamW(
            parameters,
            lr=lr,
            weight_decay=self.config.get('weight_decay', 0.01)
        )

        # Calculate total steps
        total_steps = len(self.train_loader) * self.config[f'{stage}_epochs']
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return optimizer, scheduler

    def compute_loss(self, batch):
        """Compute loss for a batch"""
        latent_vectors = batch['latent_vector'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        prompts = batch['prompt']

        # Forward pass
        outputs = self.model(
            latent_vectors=latent_vectors,
            text_prompt=prompts[0],  # Assuming same prompt for batch
            labels=input_ids
        )

        return outputs['loss']

    def validate(self):
        """Run validation"""
        self.model.eval()
        total_loss = 0
        total_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        return avg_loss

    def train_stage(self, stage: str):
        """Train a specific stage"""
        print(f"\n=== Starting {stage} training ===")

        # Set up model for this stage
        if stage == "encoder_only":
            self.model.freeze_llm()
            epochs = self.config['encoder_only_epochs']
        elif stage == "full_model":
            self.model.unfreeze_llm()
            epochs = self.config['full_model_epochs']

        optimizer, scheduler = self.create_optimizer_and_scheduler(stage)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training
            self.model.train()
            total_loss = 0
            total_batches = 0

            progress_bar = tqdm(self.train_loader, desc=f"{stage} - Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()

                loss = self.compute_loss(batch)
                loss.backward()

                # Gradient clipping
                if self.config.get('gradient_clipping', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clipping']
                    )

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                total_batches += 1

                # Update progress bar
                avg_loss = total_loss / total_batches
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

                # Log to wandb
                if self.config.get('use_wandb', False):
                    wandb.log({
                        f'{stage}_train_loss': loss.item(),
                        f'{stage}_learning_rate': scheduler.get_last_lr()[0],
                        'step': epoch * len(self.train_loader) + batch_idx
                    })

            # Validation
            val_loss = self.validate()
            print(f"Validation Loss: {val_loss:.4f}")

            # Log validation loss
            if self.config.get('use_wandb', False):
                wandb.log({
                    f'{stage}_val_loss': val_loss,
                    'epoch': epoch + 1
                })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(stage, epoch, val_loss, is_best=True)

            # Regular checkpoint
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(stage, epoch, val_loss, is_best=False)

    def save_checkpoint(self, stage: str, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        # Save path
        suffix = "best" if is_best else f"epoch_{epoch + 1}"
        path = f"checkpoints/{stage}_{suffix}.pt"

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {path}")
        return checkpoint

    def run_full_training(self):
        """Run complete two-stage training"""
        print("Starting two-stage training pipeline...")

        # Stage 1: Train only the latent encoder
        self.train_stage("encoder_only")

        # Stage 2: Fine-tune the entire model
        self.train_stage("full_model")

        print("\nTraining completed!")

        # Generate some sample outputs
        self.generate_samples()

    def generate_samples(self, num_samples: int = 5):
        """Generate sample outputs for evaluation"""
        print(f"\nGenerating {num_samples} sample outputs...")

        self.model.eval()

        with torch.no_grad():
            for i in range(min(num_samples, len(self.val_dataset))):
                sample = self.val_dataset[i]
                latent_vector = sample['latent_vector'].unsqueeze(0).to(self.device)
                prompt = sample['prompt']
                true_description = sample['description']

                # Generate
                outputs = self.model(
                    latent_vectors=latent_vector,
                    text_prompt=prompt
                )

                # Decode (simplified - you'd want beam search or sampling)
                predicted_ids = torch.argmax(outputs['logits'], dim=-1)
                predicted_text = self.model.tokenizer.decode(
                    predicted_ids[0],
                    skip_special_tokens=True
                )

                print(f"\n--- Sample {i + 1} ---")
                print(f"True: {true_description}")
                print(f"Predicted: {predicted_text}")


# Training configuration
def get_training_config():
    return {
        # Model parameters
        'model_name': 'microsoft/DialoGPT-small',
        'latent_dim': 128,
        'hidden_dim': 512,
        'num_encoder_layers': 3,

        # Training parameters
        'batch_size': 8,
        'encoder_only_epochs': 10,
        'full_model_epochs': 5,
        'encoder_lr': 1e-4,
        'full_model_lr': 5e-5,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'gradient_clipping': 1.0,

        # Other settings
        'max_length': 512,
        'num_workers': 2,
        'save_every': 2,
        'use_wandb': True,
        'project_name': 'multimodal-star-llm'
    }


# Example usage
def main():
    """Main training script"""

    # Load your data (replace with actual data loading)
    # For demonstration, creating dummy data
    num_samples = 1000
    latent_dim = 128

    # Dummy star latent vectors
    latent_vectors = np.random.randn(num_samples, latent_dim)

    # Dummy star descriptions
    descriptions = [
        f"Mass: {np.random.uniform(0.1, 50):.2f} solar masses, "
        f"Temperature: {np.random.uniform(3000, 50000):.0f} K, "
        f"Luminosity: {np.random.uniform(0.001, 1000000):.3f} solar luminosities, "
        f"Radius: {np.random.uniform(0.1, 1000):.2f} solar radii"
        for _ in range(num_samples)
    ]

    # Split data
    split_idx = int(0.8 * num_samples)
    train_latents = latent_vectors[:split_idx]
    train_descriptions = descriptions[:split_idx]
    val_latents = latent_vectors[split_idx:]
    val_descriptions = descriptions[split_idx:]

    # Get config
    config = get_training_config()

    # Create model
    from multimodal_llm import MultimodalLLM  # Import from previous artifact
    model = MultimodalLLM(
        model_name=config['model_name'],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        num_encoder_layers=config['num_encoder_layers']
    )

    # Create datasets
    train_dataset = StarDataset(
        train_latents, train_descriptions,
        model.tokenizer, config['max_length']
    )
    val_dataset = StarDataset(
        val_latents, val_descriptions,
        model.tokenizer, config['max_length']
    )

    # Create training pipeline
    trainer = TrainingPipeline(model, train_dataset, val_dataset, config)

    # Run training
    trainer.run_full_training()


if __name__ == "__main__":
    main()