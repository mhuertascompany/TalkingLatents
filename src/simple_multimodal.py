import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import json
import yaml
from pathlib import Path
import argparse
from typing import Optional
import datetime

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

# Import all necessary components
from nn.multimodal import (
    MultimodalStellarModel, 
    AlignmentConfig, 
    _load_llm_model, 
    _load_spectra_model,
    TOKENIZER_PATH,
    MODEL_PATH,
    setup
)
from data.dataset_multimodal import StellarDataset, create_stellar_dataloaders, collate_fn
from data.transforms import *
from nn.train import CLIPTrainer
from util.utils import load_checkpoints_ddp

JSON_PATH = '/data/TalkingLatents/data/dataset/stellar_descriptions.json'
FEATURES_PATH = '/data/TalkingLatents/logs/2025-07-29/features.npy'  # Optional, can be None to load all features on-the-fly

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CLIP Multimodal Stellar Model')
    
    # Data paths
    parser.add_argument('--json_file', type=str, default=JSON_PATH,
                       help='Path to stellar descriptions JSON file')
    parser.add_argument('--features_file', type=str, default=None,
                       help='Path to spectral features numpy file')
    parser.add_argument('--output_dir', type=str, default='logs',
                       help='Output directory for logs and models')
    parser.add_argument('--exp_name', type=str, default='clip_stellar',
                       help='Experiment name')
    
    # Model configuration
    parser.add_argument('--text_embedding_dim', type=int, default=2048,
                       help='LLM text embedding dimension')
    parser.add_argument('--spectral_embedding_dim', type=int, default=2048,
                       help='Spectral model embedding dimension')
    parser.add_argument('--projection_dim', type=int, default=4096,
                       help='Common projection space dimension')
    parser.add_argument('--latent_ids', type=list, nargs='*', default=['Teff', 'logg', 'FeH'],
                       help='List of latent variable IDs to include (e.g., --latent_ids mass age metallicity)')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Initial temperature for contrastive loss')
    parser.add_argument('--learnable_temperature', action='store_true',
                       help='Make temperature learnable parameter'),
    parser.add_argument('--max_seq_length', type=int, default=128,
                       help='Maximum sequence length for text inputs'),
    parser.add_argument('--project_spectral', type=bool, default=True,
                       help='Whether to project spectral embeddings to common space'),
    parser.add_argument('--checkpoint_dir', type=str, default='./logs/2025-08-27',
                          help='Directory to load model checkpoint from, if any'),
    parser.add_argument('--train', type=bool, default=False,
                          help='Whether to train the model or just evaluate'),
                    
                
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                       help='Number of warmup epochs')
    parser.add_argument('--early_stopping', type=int, default=20,
                       help='Early stopping patience')
    
    # Data splitting
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for data splitting')
    
    # Distributed training
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    # Model freezing
    parser.add_argument('--freeze_llm', action='store_true', default=True,
                       help='Freeze LLM parameters')
    parser.add_argument('--freeze_spectral', action='store_true', default=True,
                       help='Freeze spectral model parameters')
    
    # Evaluation
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--compute_retrieval_metrics', action='store_true',
                       help='Compute retrieval metrics during training')
    
    return parser.parse_args()

def create_datasets_and_loaders(args, device):
    """Create datasets and dataloaders with distributed sampling"""
    
    # Load spectral features if provided
    spectral_features = None
    if args.features_file and os.path.exists(args.features_file):
        print(f"Loading spectral features from {args.features_file}")
        spectral_features = np.load(args.features_file)
        print(f"Spectral features shape: {spectral_features.shape}")
    else:
        print("No spectral features file provided or file not found. Will use raw spectra on-the-fly.")
    
    transf = Compose([GeneralSpectrumPreprocessor(rv_norm=True), ToTensor()])

    # Create cache directory for split consistency
    cache_dir = os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create datasets for each split
    train_dataset = StellarDataset(
        json_file=args.json_file,
        features_array=spectral_features,
        spectral_transforms=transf,
        split='train',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        cache_dir=cache_dir,
        tokenizer_path=TOKENIZER_PATH,  
        max_length=args.max_seq_length
    )
    
    val_dataset = StellarDataset(
        json_file=args.json_file,
        features_array=spectral_features,
        spectral_transforms=transf,
        split='val',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        cache_dir=cache_dir,
        tokenizer_path=TOKENIZER_PATH,  
        max_length=args.max_seq_length
    )
    
    test_dataset = StellarDataset(
        json_file=args.json_file,
        features_array=spectral_features,
        spectral_transforms=transf,
        split='test',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
        cache_dir=cache_dir,
        tokenizer_path=TOKENIZER_PATH,  
        max_length=args.max_seq_length
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if dist.is_initialized() else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def create_model(args, device):
    """Create and configure the multimodal model"""
    
    print("Loading LLM model...")
    llm_model = _load_llm_model(args.batch_size, args.max_seq_length)  # Back to original
    
    print("Loading spectral model...")
    spectral_model = _load_spectra_model()
    
    # Create alignment configuration
    alignment_config = AlignmentConfig(
        text_embedding_dim=args.text_embedding_dim,
        spectral_embedding_dim=args.spectral_embedding_dim,
        latent_ids=args.latent_ids,
        projection_dim=args.projection_dim,
        text_projection_hidden_dims=[512, args.projection_dim],
        temperature=args.temperature,
        learnable_temperature=args.learnable_temperature,
        project_spectral=args.project_spectral,  # Enable spectral projection
        normalize_embeddings=True,
        symmetric_loss=True
    )
    
    # Create multimodal model
    model = MultimodalStellarModel(
        llm_model=llm_model,
        spectral_model=spectral_model,
        alignment_config=alignment_config,
        freeze_llm=args.freeze_llm,
        freeze_spectral=args.freeze_spectral,
        use_spectral_features=(args.features_file is not None and os.path.exists(args.features_file))
    )

    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        checkpoint_path = os.path.join(args.checkpoint_dir, 'clip_stellar.pth')
        model = load_checkpoints_ddp(model, checkpoint_path)
    
    # Move to device
    model = model.to(device)
    
    # Wrap with DDP if distributed training
    if dist.is_initialized():
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
        print(f"Model wrapped with DDP on device {device}")
    
    return model

def create_optimizer_and_scheduler(model, args, train_loader):
    """Create optimizer and learning rate scheduler"""
    
    # Only optimize alignment module parameters (since backbones are frozen)
    if hasattr(model, 'module'):  # DDP wrapped
        alignment_params = model.module.alignment_module.parameters()
    else:
        alignment_params = model.alignment_module.parameters()
    
    optimizer = torch.optim.AdamW(
        alignment_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler with warmup
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

def create_trainer(model, optimizer, criterion, train_loader, val_loader, device, args):
    """Create CLIP trainer"""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get world size
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    trainer = CLIPTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,  # Not used for CLIP, but required by base class
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        world_size=world_size,
        output_dim=1,  # Single accuracy metric for CLIP
        scheduler=None,  # We'll handle scheduling manually
        max_iter=np.inf,
        log_path=args.output_dir,
        exp_name=args.exp_name,
        temperature_logging=True,
        similarity_logging=True,
        retrieval_eval=args.compute_retrieval_metrics,
        retrieval_k=[1, 5, 10, 20],
        latent_ids=args.latent_ids,
    )
    
    return trainer

def save_config(args, output_dir):
    """Save training configuration"""
    config_path = os.path.join(output_dir, 'training_config.json')
    config_dict = vars(args)
    config_dict['llm_model'] = MODEL_PATH
    config_dict['tokenizer'] = TOKENIZER_PATH
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Training configuration saved to {config_path}")

def test_spectral_features(dl, model, device, num_iters=5):
    spec_model = model.module.spectral_model if hasattr(model, 'module') else model.spectral_model
    for i, batch in enumerate(dl):
        if i >= num_iters:
            break
        spectra = batch['spectra'].to(device)
        masked_spectra = batch['masked_spectra'].to(device)
        features = batch['features'].to(device)
        with torch.no_grad():
            _, spectra_pred, pred_features = spec_model(masked_spectra)
        diff = features - pred_features
        mse_features = (diff**2).mean(-1)
        mse = ((spectra - spectra_pred)**2).mean(-1)
        print(f"Batch {i}: MSE between true and predicted spectra: {mse.mean().item():.6f}")
        print(f"Batch {i}: MSE between true and predicted features: {mse_features.mean().item():.6f}")
        plt.plot(spectra[0].squeeze().cpu().numpy(), label='True Spectrum')
        plt.plot(spectra_pred[0].squeeze().cpu().numpy(), label='Predicted Spectrum')
        plt.legend()
        plt.title(f'Spectrum Prediction - Batch {i}')
        plt.savefig(f'/data/TalkingLatents/figs/spectrum_prediction_batch_{i}.png')
        plt.close()

        plt.plot(features[0].squeeze().cpu().numpy() - pred_features[0].squeeze().cpu().numpy() , label='features diff')
        plt.legend()
        plt.title(f'features Prediction - Batch {i}')
        plt.savefig(f'/data/TalkingLatents/figs/features_comparison_batch_{i}.png')
        plt.close()

def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_args()
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    args.output_dir = os.path.join(args.output_dir, date)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup distributed training
    local_rank, world_size, gpus_per_node = setup()
    device = torch.cuda.current_device()
    
    print(f"Distributed training setup complete. Local rank: {local_rank}, World size: {world_size}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    
    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_loader, val_loader, test_loader = create_datasets_and_loaders(args, device)
    
    # Create model
    print("Creating multimodal model...")
    model = create_model(args, device)

    if args.features_file is not None and os.path.exists(args.features_file):
        print("Testing spectral features prediction...")
        test_spectral_features(train_loader, model, device, num_iters=5)
    
    # Create optimizer and scheduler
    print("Creating optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, train_loader)
    
    # Dummy criterion (not used for CLIP but required by base trainer)
    criterion = nn.MSELoss()
    
    # Create trainer
    print("Creating CLIP trainer...")
    trainer = create_trainer(model, optimizer, criterion, train_loader, val_loader, device, args)
    trainer.scheduler = scheduler  # Assign scheduler after trainer creation
    

    # Save configuration
    if local_rank == 0:  # Only save on main process
        save_config(args, args.output_dir)
    
    # Print training info
    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTraining Configuration:")
        print(f"  Experiment: {args.exp_name}")
        print(f"  Batch size: {args.batch_size} per GPU ({args.batch_size * world_size} total)")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"  Output directory: {args.output_dir}")
        print("-" * 60)
    
    # Train the model
    if args.train:
        print("Starting training...")
        results = trainer.fit(
            num_epochs=args.num_epochs,
            device=device,
            early_stopping=args.early_stopping,
            best='loss'  # Use loss for early stopping
        )
    
    # Run final evaluation on test set
    if local_rank == 0:
        print("Running final evaluation on test set...")
        test_results = trainer.predict(test_loader, device, load_best=True, compute_embeddings=True)
        
        # Save test results
        test_results_path = os.path.join(args.output_dir, f'{args.exp_name}_test_results.json')
        
        # Convert tensors to lists for JSON serialization
        test_results_json = {
            'avg_loss': test_results['avg_loss'],
            'avg_accuracy': test_results['avg_accuracy'],
            'losses': test_results['losses'],
            'accuracies': test_results['accuracies'],
            'num_samples': len(test_results['descriptions'])
        }
        
        # Add retrieval metrics if computed
        if 'retrieval_metrics' in test_results:
            test_results_json['retrieval_metrics'] = test_results['retrieval_metrics']
        
        with open(test_results_path, 'w') as f:
            json.dump(test_results_json, f, indent=2)
        
        print(f"Test results saved to {test_results_path}")
        
        # Save embeddings separately (they're too large for JSON)
        if 'text_embeddings' in test_results:
            embeddings_path = os.path.join(args.output_dir, f'{args.exp_name}_test_embeddings.npz')
            np.savez(embeddings_path,
                    text_embeddings=test_results['text_embeddings'].numpy(),
                    spectral_embeddings=test_results['spectral_embeddings'].numpy(),
                    similarities=test_results['similarities'].numpy())
            print(f"Test embeddings saved to {embeddings_path}")
        
        print("\nTraining completed successfully!")
        print(f"Final test loss: {test_results['avg_loss']:.6f}")
        print(f"Final test accuracy: {test_results['avg_accuracy']:.4f}")
        
        if 'retrieval_metrics' in test_results:
            print("\nRetrieval Metrics:")
            for metric, value in test_results['retrieval_metrics'].items():
                if 'recall' in metric:
                    print(f"  {metric}: {value:.4f}")


def run_training_with_error_handling():
    """Wrapper to handle training with error recovery"""
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise


if __name__ == "__main__":
    """
    Example usage:
    
    # Single GPU training
    python clip_training_pipeline.py \
        --json_file data/stellar_descriptions.json \
        --features_file data/spectral_features.npy \
        --output_dir logs/clip_experiment \
        --exp_name stellar_clip_v1 \
        --batch_size 32 \
        --num_epochs 100 \
        --learning_rate 1e-4 \
        --projection_dim 256 \
        --learnable_temperature \
        --compute_retrieval_metrics
    
    # Multi-GPU SLURM training
    srun python clip_training_pipeline.py \
        --json_file data/stellar_descriptions.json \
        --features_file data/spectral_features.npy \
        --output_dir logs/clip_experiment \
        --exp_name stellar_clip_distributed \
        --batch_size 16 \
        --num_epochs 200 \
        --learning_rate 5e-4 \
        --projection_dim 512 \
        --learnable_temperature \
        --compute_retrieval_metrics
    """
    
    print("=" * 80)
    print("CLIP MULTIMODAL STELLAR MODEL TRAINING")
    print("=" * 80)
    
    run_training_with_error_handling()


