import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from torch.cuda.amp import autocast
import gc

# Import your existing models
from nn.llm import MultimodalLlamaModel
from nn.models import MultiTaskRegressor
from llama3.llama.model import Transformer, ModelArgs
from util.utils import Container
import yaml


class MultimodalLLaMA(pl.LightningModule):
    """
    PyTorch Lightning module for multimodal LLaMA training
    """
    
    def __init__(
        self,
        llm_model_path: str,
        spectral_config_path: str,
        spectral_weights_path: str,
        tokenizer_path: str,
        load_spectral_model: bool = True,
        spectral_embedding_dim: int = 2048,
        hidden_dim: int = 512,
        num_spectral_features: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.001,
        warmup_epochs: int = 2,
        max_epochs: int = 100,
        freeze_llm: bool = True,
        freeze_spectral: bool = True,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        max_seq_length: int = 128,
        batch_size: int = 32,  # For scheduler calculations
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Store important params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.freeze_llm = freeze_llm
        self.freeze_spectral = freeze_spectral
        self.use_lora = use_lora
        self.lora_config = lora_config or {}
        self.batch_size = batch_size
        self.load_spectral_model = load_spectral_model
        
        # Initialize models
        self.setup_models(
            llm_model_path=llm_model_path,
            spectral_config_path=spectral_config_path,
            spectral_weights_path=spectral_weights_path,
            tokenizer_path=tokenizer_path,
            spectral_embedding_dim=spectral_embedding_dim,
            hidden_dim=hidden_dim,
            num_spectral_features=num_spectral_features,
            max_seq_length=max_seq_length
        )
        
        # Validation metrics storage
        self.validation_outputs = []
        
        # For generation during validation
        self.tokenizer = None
        self._load_tokenizer(tokenizer_path)
    
    def setup_models(self, **kwargs):
        """Setup the multimodal model components"""
        
        # Load LLaMA model
        llm_model = self._load_llama_model(
            kwargs['llm_model_path'], 
            kwargs['max_seq_length']
        )
        
        if self.load_spectral_model:
            # Load spectral model
            spectral_model = self._load_spectral_model(
                kwargs['spectral_config_path'],
                kwargs['spectral_weights_path']
            )
        else:
            spectral_model = None
        
        # Create multimodal wrapper
        self.model = MultimodalLlamaModel(
            base_model=llm_model,
            fm_model=spectral_model,
            latent_dim=kwargs['spectral_embedding_dim'],
            hidden_dim=kwargs['hidden_dim'],
            num_spectral_features=kwargs['num_spectral_features']
        )
        
        # Apply freezing
        self._apply_freezing()
        
        # Apply LoRA if requested
        if self.use_lora:
            self._apply_lora()
    
    def _load_llama_model(self, model_path: str, max_seq_length: int) -> Transformer:
        """Load LLaMA model with FairScale setup"""
        
        # Initialize FairScale for single GPU (Lightning handles multi-GPU)
        import fairscale.nn.model_parallel.initialize as fs_init
        import torch.distributed as dist
        
        if not dist.is_initialized():
            # Lightning will handle distributed setup, but we need minimal setup for FairScale
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12355")
            dist.init_process_group("gloo", rank=0, world_size=1)
        
        if not fs_init.model_parallel_is_initialized():
            fs_init.initialize_model_parallel(1)
        
        # Load model parameters
        with open(Path(model_path) / "params.json", "r") as f:
            params = json.loads(f.read())
        
        model_args = ModelArgs(
            max_batch_size=self.batch_size,
            max_seq_len=max_seq_length,
            **params,
        )
        
        # Create model
        model = Transformer(model_args)
        
        # Load checkpoint
        checkpoints = sorted(Path(model_path).glob("*.pth"))
        if checkpoints:
            checkpoint = torch.load(checkpoints[0], map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded LLaMA checkpoint: {checkpoints[0]}")
        
        return model
    
    def _load_spectral_model(self, config_path: str, weights_path: str) -> MultiTaskRegressor:
        """Load spectral model"""
        
        # Load config
        config = yaml.safe_load(open(config_path, 'r'))
        config['model_args']['avg_output'] = False
        
        # Create model
        model = MultiTaskRegressor(
            Container(**config['model_args']), 
            Container(**config['conformer_args'])
        )
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Remove 'module.' prefix if present
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def _load_tokenizer(self, tokenizer_path: str):
        """Load tokenizer for generation"""
        try:
            from llama3.llama.tokenizer import Tokenizer
            self.tokenizer = Tokenizer(model_path=tokenizer_path)
            print(f"Loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            self.tokenizer = None
    
    def _apply_freezing(self):
        """Apply parameter freezing"""
        if self.freeze_llm:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            print("Froze LLM parameters")
        
        if self.freeze_spectral:
            for param in self.model.fm_model.parameters():
                param.requires_grad = False
            print("Froze spectral model parameters")
    
    def _apply_lora(self):
        """Apply LoRA if configured"""
        if self.lora_config:
            from nn.state_space_llm import apply_lora_to_model
            
            # This would need to be adapted based on your LoRA implementation
            target_modules = self.lora_config.get('target_modules', [])
            rank = self.lora_config.get('rank', 16)
            alpha = self.lora_config.get('alpha', 32)
            dropout = self.lora_config.get('dropout', 0.1)
            
            # Apply LoRA to specified modules
            # You'll need to adapt this based on your existing LoRA code
            print("LoRA application would go here - adapt from your existing code")
    
    def get_loss(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Compute language modeling loss"""
        # Shift for autoregressive prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        
        # Only compute loss on actual answer tokens (not -100)
        active_mask = (shift_labels != -100)
        
        if not active_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = active_mask.view(-1)
        
        # Compute loss only on answer tokens
        active_logits = flat_logits[flat_mask]
        active_labels = flat_labels[flat_mask]
        
        loss = F.cross_entropy(active_logits, active_labels, reduction='mean')
        return loss
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        input_ids = batch['input_ids']
        input_spectra = batch['masked_spectra']
        special_token_positions = batch['feature_start_indices']
        
        outputs = self.model(
            input_ids=input_ids,
            input_spectra=input_spectra,
            special_token_positions=special_token_positions,
        )
        
        return outputs
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        
        # Forward pass
        outputs = self(batch)
        logits = outputs['logits']
        
        # Compute loss
        target_ids = batch['target_ids']
        loss = self.get_loss(logits, target_ids)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        # Log memory usage periodically
        if batch_idx % 100 == 0:
            self._log_memory_usage('train')
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        
        # Forward pass
        outputs = self(batch)
        logits = outputs['logits']
        
        # Compute loss
        target_ids = batch['target_ids']
        loss = self.get_loss(logits, target_ids)
        
        # Store for epoch-end processing
        self.validation_outputs.append({
            'loss': loss,
            'batch': batch,
            'outputs': outputs
        })
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        """Process validation outputs at epoch end"""
        
        # Generate sample responses (only on rank 0)
        if self.trainer.is_global_zero and len(self.validation_outputs) > 0:
            self._generate_validation_samples()
        
        # Clear validation outputs
        self.validation_outputs.clear()
    
    @rank_zero_only
    def _generate_validation_samples(self, num_samples: int = 3):
        """Generate sample responses for validation monitoring"""
        
        if self.tokenizer is None:
            return
        
        print(f"\n{'='*80}")
        print(f"VALIDATION SAMPLE EVALUATION - EPOCH {self.current_epoch}")
        print(f"{'='*80}")
        
        sample_count = 0
        for val_output in self.validation_outputs[:num_samples]:
            if sample_count >= num_samples:
                break
                
            batch = val_output['batch']
            
            # Generate response for first sample in batch
            try:
                generated_text, input_text, target_text, _ = self.model.generate_response_from_batch(
                    batch_data=batch,
                    batch_idx=0,
                    tokenizer=self.tokenizer,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9
                )
                
                obsid = batch['obsids'][0] if 'obsids' in batch else "Unknown"
                
                print(f"\n{'-'*60}")
                print(f"SAMPLE {sample_count + 1} (OBSID: {obsid})")
                print(f"{'-'*60}")
                print(f"QUESTION: {input_text}")
                print(f"TRUE ANSWER: {target_text}")
                print(f"GENERATED ANSWER: {generated_text}")
                
                sample_count += 1
                
            except Exception as e:
                print(f"Error generating sample {sample_count}: {e}")
                continue
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        
        # Get trainable parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        if not trainable_params:
            raise ValueError("No trainable parameters found!")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Create scheduler
        # Lightning handles steps_per_epoch automatically
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_epochs / self.max_epochs)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_before_optimizer_step(self, optimizer):
        """Called before optimizer step - good place for gradient clipping"""
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    
    @rank_zero_only
    def _log_memory_usage(self, stage: str):
        """Log memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            self.log(f'{stage}/memory_allocated_gb', allocated)
            self.log(f'{stage}/memory_reserved_gb', reserved)
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch"""
        # Clear cache periodically
        if self.current_epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Prediction step for inference"""
        
        outputs = self(batch)
        
        # Generate text if tokenizer is available
        generated_responses = []
        if self.tokenizer is not None:
            batch_size = batch['input_ids'].shape[0]
            for i in range(batch_size):
                try:
                    generated_text, input_text, target_text, _ = self.model.generate_response_from_batch(
                        batch_data=batch,
                        batch_idx=i,
                        tokenizer=self.tokenizer,
                        max_new_tokens=100,
                        temperature=0.7,
                        top_p=0.9
                    )
                    generated_responses.append({
                        'generated': generated_text,
                        'input': input_text,
                        'target': target_text
                    })
                except Exception as e:
                    generated_responses.append({
                        'generated': f"Error: {e}",
                        'input': "Unknown",
                        'target': "Unknown"
                    })
        
        return {
            'logits': outputs['logits'],
            'generated_responses': generated_responses,
            'obsids': batch.get('obsids', [])
        }