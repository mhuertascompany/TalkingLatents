#!/usr/bin/env python3

import os
import json
import argparse
import datetime
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.profilers import PyTorchProfiler

import os
os.system('pip install tiktoken fairscale fire blobfile pytorch-lightning')
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

# Import your Lightning modules
from nn.llm_pl import MultimodalLLaMA
from data.dataset_pl import StellarDataModule

# Constants (update these paths for your setup)
JSON_PATH = '/data/TalkingLatents/data/dataset/stellar_descriptions_questions_short.json'
FEATURES_PATH = '/data/TalkingLatents/logs/2025-07-29/features.npy'
MODEL_PATH = "/data/.llama/Llama3.1-8B"
TOKENIZER_PATH = "/data/.llama/Llama3.1-8B/tokenizer.model"
SPECTRA_CONFIG_PATH = "/data/DESA/logs/spec_decode2_2025-02-16/MultiTaskRegressor_spectra__decode_4_complete_config.yaml"
SPECTRA_WEIGHTS_PATH = "/data/DESA/logs/spec_decode2_2025-02-16/MultiTaskRegressor_spectra_decode_4.pth"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Multimodal LLaMA with PyTorch Lightning')
    
    # Data arguments
    parser.add_argument('--json_file', type=str, default=JSON_PATH,
                       help='Path to stellar descriptions JSON file')
    parser.add_argument('--features_file', type=str, default=FEATURES_PATH,
                       help='Path to spectral features numpy file (optional)')
    parser.add_argument('--output_dir', type=str, default='lightning_logs',
                       help='Output directory for logs and checkpoints')
    parser.add_argument('--exp_name', type=str, default='multimodal_llama',
                       help='Experiment name')
    
    # Model arguments
    parser.add_argument('--llm_model_path', type=str, default=MODEL_PATH,
                       help='Path to LLaMA model directory')
    parser.add_argument('--tokenizer_path', type=str, default=TOKENIZER_PATH,
                       help='Path to tokenizer file')
    parser.add_argument('--spectral_config_path', type=str, default=SPECTRA_CONFIG_PATH,
                       help='Path to spectral model config')
    parser.add_argument('--spectral_weights_path', type=str, default=SPECTRA_WEIGHTS_PATH,
                       help='Path to spectral model weights')
    parser.add_argument('--spectral_embedding_dim', type=int, default=2048,
                       help='Spectral model embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for projections')
    parser.add_argument('--num_spectral_features', type=int, default=1,
                       help='Number of spectral features to integrate')
    parser.add_argument('--max_seq_length', type=int, default=128,
                       help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                       help='Number of warmup epochs')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value')
    
    # Data splitting
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed')
    
    # Hardware/performance arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--precision', type=str, default='16-mixed',
                       choices=['32', '16-mixed', 'bf16-mixed'],
                       help='Training precision')
    parser.add_argument('--strategy', type=str, default='auto',
                       choices=['auto', 'ddp', 'ddp_find_unused_parameters_true', 'fsdp'],
                       help='Distributed strategy')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # Model freezing and LoRA
    parser.add_argument('--freeze_llm', action='store_true', default=True,
                       help='Freeze LLM parameters')
    parser.add_argument('--freeze_spectral', action='store_true', default=True,
                       help='Freeze spectral model parameters')
    parser.add_argument('--use_lora', action='store_true', default=False,
                       help='Use LoRA for parameter-efficient fine-tuning')
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')
    
    # Callbacks and monitoring
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--save_top_k', type=int, default=3,
                       help='Save top k checkpoints')
    parser.add_argument('--monitor_metric', type=str, default='val/loss',
                       help='Metric to monitor for checkpointing')
    parser.add_argument('--monitor_mode', type=str, default='min',
                       choices=['min', 'max'],
                       help='Mode for monitoring metric')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='multimodal-llama',
                       help='W&B project name')
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                       help='Log every N steps')
    
    # Testing and inference
    parser.add_argument('--test_only', action='store_true', default=False,
                       help='Run testing only (no training)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint for testing/inference')
    
    # Profiling and debugging
    parser.add_argument('--profile', action='store_true', default=False,
                       help='Enable profiling')
    parser.add_argument('--detect_anomaly', action='store_true', default=False,
                       help='Enable anomaly detection for debugging')
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                       help='Limit training batches (for debugging)')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                       help='Limit validation batches (for debugging)')
    
    return parser.parse_args()


def setup_callbacks(args):
    """Setup training callbacks"""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename=f'{args.exp_name}-{{epoch:02d}}-{{val/loss:.2f}}',
        monitor=args.monitor_metric,
        mode=args.monitor_mode,
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=args.monitor_metric,
        mode=args.monitor_mode,
        patience=args.early_stopping_patience,
        verbose=True,
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Rich progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    # Rich model summary
    model_summary = RichModelSummary(max_depth=2)
    callbacks.append(model_summary)
    
    return callbacks


def setup_logger(args):
    """Setup experiment logger"""
    
    # Create experiment directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(args.output_dir, f"{args.exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    if args.use_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=f"{args.exp_name}_{timestamp}",
            save_dir=exp_dir,
            log_model=True
        )
    else:
        logger = TensorBoardLogger(
            save_dir=exp_dir,
            name="tensorboard_logs",
            log_graph=True
        )
    
    return logger, exp_dir


def setup_strategy(args):
    """Setup distributed training strategy"""
    
    if args.strategy == 'auto':
        # Let Lightning decide the best strategy
        return 'auto'
    elif args.strategy == 'ddp':
        return DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    elif args.strategy == 'ddp_find_unused_parameters_true':
        return DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    elif args.strategy == 'fsdp':
        from pytorch_lightning.strategies import FSDPStrategy
        return FSDPStrategy(
            auto_wrap_policy=None,  # You may need to customize this
        )
    else:
        return args.strategy


def create_trainer(args, callbacks, logger, strategy):
    """Create PyTorch Lightning trainer"""
    
    # Setup profiler if requested
    profiler = None
    if args.profile:
        profiler = PyTorchProfiler(
            dirpath=os.path.join(args.output_dir, 'profiler'),
            filename="perf_logs",
            export_to_chrome=True,
        )
    
    # Detect SLURM environment
    plugins = []
    if 'SLURM_JOB_ID' in os.environ:
        plugins.append(SLURMEnvironment())
        print("Detected SLURM environment")
    
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        precision=args.precision,
        strategy=strategy,
        devices='auto',  # Use all available GPUs
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        profiler=profiler,
        plugins=plugins,
        # Debugging options
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        detect_anomaly=args.detect_anomaly,
        # Performance optimizations
        num_sanity_val_steps=2,
        sync_batchnorm=True if strategy != 'auto' else False,
    )
    
    return trainer


def save_config(args, exp_dir):
    """Save experiment configuration"""
    config_path = os.path.join(exp_dir, 'config.json')
    
    config_dict = vars(args).copy()
    # Convert Path objects to strings for JSON serialization
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to {config_path}")


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    pl.seed_everything(args.random_seed, workers=True)
    
    # Enable anomaly detection if requested
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    
    # Setup experiment logging
    logger, exp_dir = setup_logger(args)
    print(f"Experiment directory: {exp_dir}")
    
    # Save configuration
    save_config(args, exp_dir)
    
    # Setup LoRA configuration
    lora_config = None
    if args.use_lora:
        lora_config = {
            'rank': args.lora_rank,
            'alpha': args.lora_alpha,
            'dropout': args.lora_dropout,
            'target_modules': ['wq', 'wk', 'wv', 'wo'],  # Adjust based on your model
        }
    
    # Create data module
    print("Creating data module...")
    data_module = StellarDataModule(
        json_file=args.json_file,
        tokenizer_path=args.tokenizer_path,
        features_file=args.features_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        max_seq_length=args.max_seq_length,
        num_spectral_features=args.num_spectral_features,
        cache_dir=os.path.join(exp_dir, 'cache')
    )
    
    # Create model
    print("Creating Lightning module...")
    model = MultimodalLLaMA(
        llm_model_path=args.llm_model_path,
        spectral_config_path=args.spectral_config_path,
        spectral_weights_path=args.spectral_weights_path,
        load_spectral_model=args.features_file is not None,
        tokenizer_path=args.tokenizer_path,
        spectral_embedding_dim=args.spectral_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_spectral_features=args.num_spectral_features,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.num_epochs,
        freeze_llm=args.freeze_llm,
        freeze_spectral=args.freeze_spectral,
        use_lora=args.use_lora,
        lora_config=lora_config,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )
    
    # Setup callbacks and strategy
    callbacks = setup_callbacks(args)
    strategy = setup_strategy(args)
    
    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(args, callbacks, logger, strategy)
    
    # Print training info
    print(f"\nTraining Configuration:")
    print(f"  Experiment: {args.exp_name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Precision: {args.precision}")
    print(f"  Strategy: {strategy}")
    print(f"  Devices: {trainer.num_devices}")
    print(f"  Freeze LLM: {args.freeze_llm}")
    print(f"  Freeze Spectral: {args.freeze_spectral}")
    print(f"  Use LoRA: {args.use_lora}")
    print("-" * 60)
    
    if not args.test_only:
        # Train the model
        print("Starting training...")
        trainer.fit(model, datamodule=data_module)
        
        # Test the best model
        print("Testing best model...")
        trainer.test(model, datamodule=data_module, ckpt_path="best")
        
    else:
        # Test only mode
        if args.checkpoint_path:
            print(f"Testing model from checkpoint: {args.checkpoint_path}")
            trainer.test(model, datamodule=data_module, ckpt_path=args.checkpoint_path)
        else:
            print("Error: checkpoint_path required for test_only mode")
            return
    
    print("Training completed successfully!")


if __name__ == "__main__":
    """
    Example usage:
    
    # Single GPU training
    python lightning_training.py \
        --exp_name stellar_llama_v1 \
        --batch_size 4 \
        --num_epochs 50 \
        --learning_rate 1e-4 \
        --precision 16-mixed \
        --use_wandb
    
    # Multi-GPU SLURM training
    srun python lightning_training.py \
        --exp_name stellar_llama_distributed \
        --batch_size 2 \
        --num_epochs 100 \
        --learning_rate 5e-5 \
        --precision 16-mixed \
        --strategy ddp \
        --num_workers 4 \
        --use_wandb
    
    # Test only
    python lightning_training.py \
        --test_only \
        --checkpoint_path path/to/checkpoint.ckpt \
        --batch_size 8
    """
    
    print("=" * 80)
    print("MULTIMODAL LLAMA TRAINING WITH PYTORCH LIGHTNING")
    print("=" * 80)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise