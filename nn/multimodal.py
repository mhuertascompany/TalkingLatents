import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Optional, Tuple, Any, List
import math
from pathlib import Path
import json
import yaml
from dataclasses import dataclass, field
import numpy as np
import os
os.system('pip install tiktoken fairscale fire blobfile')
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from llama3.llama.model import Transformer, ModelArgs
from nn.models import MultiTaskRegressor
from util.utils import *

TOKENIZER_PATH = "/data/.llama/Llama3.2-1B/tokenizer.model"
MODEL_PATH = "/data/.llama/Llama3.2-1B"
SPECTRA_CONFIG_PATH = "/data/DESA/logs/spec_decode2_2025-02-16/MultiTaskRegressor_spectra__decode_4_complete_config.yaml"
SPECTRA_WEIGHTS_PATH = "/data/DESA/logs/spec_decode2_2025-02-16/MultiTaskRegressor_spectra_decode_4.pth"

@dataclass
class AlignmentConfig:
    """Configuration for the alignment module"""
    
    # Projection dimensions
    text_embedding_dim: int = 768  # LLM output dimension
    spectral_embedding_dim: int = 512  # Spectral model output dimension  
    projection_dim: int = 256  # Common embedding space dimension
    latent_ids: List[str] = field(default_factory=lambda: [])  # List of latent variable names to concatenate
    
    # MLP projection network
    text_projection_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    text_projection_dropout: float = 0.1
    text_projection_activation: str = 'gelu'  # 'relu', 'gelu', 'tanh'
    text_projection_layer_norm: bool = True
    
    # Spectral projection (optional, if spectral embeddings need projection too)
    project_spectral: bool = False
    spectral_projection_hidden_dims: List[int] = field(default_factory=lambda: [256])
    spectral_projection_dropout: float = 0.1
    
    # Temperature scaling for contrastive loss
    temperature: float = 0.07
    learnable_temperature: bool = True
    temperature_init: float = 0.07
    temperature_min: float = 0.01
    temperature_max: float = 1.0
    
    # Loss configuration
    loss_type: str = 'clip'  # 'clip', 'simclr', 'infonce'
    symmetric_loss: bool = True  # Compute loss in both directions (text->spectral, spectral->text)
    
    # Normalization
    normalize_embeddings: bool = True
    
    # Additional regularization
    embedding_dropout: float = 0.0
    

class MLPProjection(nn.Module):
    """MLP projection head for embedding alignment"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int],
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 layer_norm: bool = True):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layer_modules = []
            
            # Linear layer
            layer_modules.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Layer norm (except last layer)
            if layer_norm and i < len(dims) - 2:
                layer_modules.append(nn.LayerNorm(dims[i + 1]))
            
            # Activation (except last layer)
            if i < len(dims) - 2:
                if activation == 'relu':
                    layer_modules.append(nn.ReLU())
                elif activation == 'gelu':
                    layer_modules.append(nn.GELU())
                elif activation == 'tanh':
                    layer_modules.append(nn.Tanh())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
                
                # Dropout
                if dropout > 0:
                    layer_modules.append(nn.Dropout(dropout))
            
            self.layers.append(nn.Sequential(*layer_modules))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class AlignmentModule(nn.Module):
    """CLIP-style alignment module for text and spectral embeddings"""
    
    def __init__(self, config: AlignmentConfig):
        super().__init__()
        self.config = config
        self.project_spectral = config.project_spectral
        
        # Text projection network
        self.text_projection = MLPProjection(
            input_dim=config.text_embedding_dim + len(config.latent_ids),
            output_dim=config.projection_dim,
            hidden_dims=config.text_projection_hidden_dims,
            dropout=config.text_projection_dropout,
            activation=config.text_projection_activation,
            layer_norm=config.text_projection_layer_norm
        )
        
        # Spectral projection network (optional)
        if config.project_spectral:
            self.spectral_projection = MLPProjection(
                input_dim=config.spectral_embedding_dim,
                output_dim=config.projection_dim,
                hidden_dims=config.spectral_projection_hidden_dims,
                dropout=config.spectral_projection_dropout,
                activation=config.text_projection_activation,
                layer_norm=config.text_projection_layer_norm
            )

        # else:
        #     # Simple linear projection to match dimensions
        #     self.spectral_projection = nn.Linear(config.spectral_embedding_dim, config.projection_dim)
        
        # Temperature parameter for contrastive loss
        if config.learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(config.temperature_init)))
        else:
            self.register_buffer('temperature', torch.tensor(config.temperature))
        
        # Embedding dropout
        if config.embedding_dropout > 0:
            self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        else:
            self.embedding_dropout = nn.Identity()
    
    def get_temperature(self) -> torch.Tensor:
        """Get current temperature value"""
        if self.config.learnable_temperature:
            # Clamp temperature to reasonable range
            temp = torch.exp(self.log_temperature)
            return torch.clamp(temp, self.config.temperature_min, self.config.temperature_max)
        else:
            return self.temperature
    
    def forward(self, 
                text_embeddings: torch.Tensor, 
                spectral_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through alignment module
        
        Args:
            text_embeddings: [batch_size, text_embedding_dim]
            spectral_embeddings: [batch_size, spectral_embedding_dim]
            
        Returns:
            Dict containing projected embeddings and similarities
        """
        batch_size = text_embeddings.size(0)
        
        # Project embeddings
        text_projected = self.text_projection(text_embeddings)
        if self.project_spectral:
            spectral_projected = self.spectral_projection(spectral_embeddings)
        else:
            spectral_projected = spectral_embeddings # Assume already in correct dim
        
        # Apply dropout
        text_projected = self.embedding_dropout(text_projected)
        spectral_projected = self.embedding_dropout(spectral_projected)
        
        # Normalize embeddings if requested
        if self.config.normalize_embeddings:
            text_projected = F.normalize(text_projected, p=2, dim=1)
            spectral_projected = F.normalize(spectral_projected, p=2, dim=1)
        
        # Compute similarity matrix
        temperature = self.get_temperature()
        similarity_matrix = torch.matmul(text_projected, spectral_projected.T) / temperature
        
        return {
            'text_embeddings': text_projected,
            'spectral_embeddings': spectral_projected,
            'similarity_matrix': similarity_matrix,
            'temperature': temperature
        }


class MultimodalStellarModel(nn.Module):
    """
    Multimodal stellar model combining LLM and spectral foundation model
    with CLIP-style alignment
    """
    
    def __init__(self,
                 llm_model: nn.Module,
                 spectral_model: nn.Module,
                 alignment_config: AlignmentConfig,
                 freeze_llm: bool = True,
                 freeze_spectral: bool = True,
                 use_spectral_features: bool = False):
        """
        Args:
            llm_model: Pre-trained language model that outputs embeddings
            spectral_model: Pre-trained spectral foundation model  
            alignment_config: Configuration for the alignment module
            freeze_llm: Whether to freeze LLM parameters
            freeze_spectral: Whether to freeze spectral model parameters
        """
        super().__init__()
        
        self.llm_model = llm_model
        self.spectral_model = spectral_model
        self.alignment_config = alignment_config
        self.use_spectral_features = use_spectral_features
        
        # Freeze models if requested
        if freeze_llm:
            for param in self.llm_model.parameters():
                param.requires_grad = False
            print("LLM parameters frozen")
            
        if freeze_spectral:
            for param in self.spectral_model.parameters():
                param.requires_grad = False
            print("Spectral model parameters frozen")
        
        # Create alignment module
        self.alignment_module = AlignmentModule(alignment_config)
        
        print(f"Multimodal model created with projection dim: {alignment_config.projection_dim}")
    
    def get_text_embeddings(self, text_inputs: Any) -> torch.Tensor:
        """
        Extract embeddings from text using LLM
        text_inputs should be token IDs tensor for LLaMA: [batch_size, seq_len]
        """
        with torch.no_grad():
            # text_inputs is [batch_size, seq_len] token IDs
            output, h = self.llm_model(text_inputs, start_pos=0)  # [batch_size, seq_len, vocab_size]
        
            # Use mean pooling over sequence dimension  
            return h.mean(dim=1)  # [batch_size, vocab_size]
    
    def get_spectral_embeddings(self, spectral_inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from spectral data using foundation model
        """
        if self.use_spectral_features:
            return spectral_inputs  # Assume pre-extracted features
        with torch.no_grad() if not self.training else torch.enable_grad():
            output_reg, output_dec, x_enc = self.spectral_model(spectral_inputs)
            return x_enc  # [batch_size, spectral_embedding_dim]
     
    def forward(self, 
                text_inputs: Any,
                spectral_inputs: torch.Tensor,
                latent: Any = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multimodal model
        
        Args:
            text_inputs: Text inputs for LLM (format depends on LLM)
            spectral_inputs: Spectral data [batch_size, spectral_features]
            
        Returns:
            Dict with embeddings, similarities, and alignment outputs
        """
        # Extract embeddings from both modalities
        text_embeddings = self.get_text_embeddings(text_inputs)
        spectral_embeddings = self.get_spectral_embeddings(spectral_inputs)
        
        # Ensure embeddings are 2D
        if len(text_embeddings.shape) == 1:
            text_embeddings = text_embeddings.unsqueeze(0)
        if len(spectral_embeddings.shape) == 1:
            spectral_embeddings = spectral_embeddings.unsqueeze(0)
        
        if latent is not None:
            # Concatenate latent variables if provided
            text_embeddings = torch.cat([text_embeddings, latent], dim=1)

        # print("text_embeddings shape: ", text_embeddings.shape, 
        #       " spectral_embeddings shape: ", spectral_embeddings.shape)
        # print("text example: ", text_embeddings[0, :10],
        #       " spectral example: ", spectral_embeddings[0, :10])
        
        # Align embeddings
        alignment_outputs = self.alignment_module(text_embeddings, spectral_embeddings)
        
        # Combine all outputs
        return {
            'raw_text_embeddings': text_embeddings,
            'raw_spectral_embeddings': spectral_embeddings,
            **alignment_outputs
        }
    
    def compute_contrastive_loss(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute CLIP-style contrastive loss
        
        Args:
            similarity_matrix: [batch_size, batch_size] similarity matrix
            
        Returns:
            Contrastive loss
        """
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Text-to-spectral loss
        text_to_spectral_loss = F.cross_entropy(similarity_matrix, labels)
        
        if self.alignment_config.symmetric_loss and self.alignment_config.project_spectral:
            # Spectral-to-text loss
            spectral_to_text_loss = F.cross_entropy(similarity_matrix.T, labels)
            return (text_to_spectral_loss + spectral_to_text_loss) / 2
        else:
            return text_to_spectral_loss
    
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Training step with contrastive loss
        """
        # Extract inputs
        descriptions = batch['descriptions']
        features = batch['features']  # [batch_size, feature_dim]
        
        # Forward pass
        outputs = self.forward(descriptions, features)
        
        # Compute loss
        loss = self.compute_contrastive_loss(outputs['similarity_matrix'])
        
        return {
            'loss': loss,
            'temperature': outputs['temperature'],
            **outputs
        }
    
    def get_retrieval_scores(self, 
                           text_queries: List[str], 
                           spectral_database: torch.Tensor) -> torch.Tensor:
        """
        Compute retrieval scores between text queries and spectral database
        
        Args:
            text_queries: List of text descriptions
            spectral_database: [n_spectra, spectral_dim] tensor of spectral features
            
        Returns:
            [len(text_queries), n_spectra] similarity scores
        """
        self.eval()
        with torch.no_grad():
            # Get text embeddings
            text_embeddings = self.get_text_embeddings(text_queries)
            text_projected = self.alignment_module.text_projection(text_embeddings)
            
            # Get spectral embeddings
            spectral_embeddings = self.get_spectral_embeddings(spectral_database)
            spectral_projected = self.alignment_module.spectral_projection(spectral_embeddings)
            
            # Normalize if needed
            if self.alignment_config.normalize_embeddings:
                text_projected = F.normalize(text_projected, p=2, dim=1)
                spectral_projected = F.normalize(spectral_projected, p=2, dim=1)
            
            # Compute similarities
            similarities = torch.matmul(text_projected, spectral_projected.T)
            
        return similarities


# Training utilities
def create_trainer(model: MultimodalStellarModel,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  learning_rate: float = 1e-4,
                  weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """Create optimizer for training"""
    
    # Only optimize alignment module parameters (frozen models don't contribute gradients)
    optimizer = torch.optim.AdamW(
        model.alignment_module.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    return optimizer

def _load_llm_model(max_batch_size=32, max_seq_len=512) -> Transformer:
    """Load the base LLaMA model"""
    with open(Path(MODEL_PATH) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
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

def _load_spectra_model():
    config = yaml.safe_load(open(SPECTRA_CONFIG_PATH, 'r'))
    config['model_args']['avg_output'] = False
    model = MultiTaskRegressor(Container(**config['model_args']), Container(**config['conformer_args']))
    model = load_checkpoints_ddp(model, SPECTRA_WEIGHTS_PATH)

    return model

def setup():
    """
    Setup the distributed training environment with fairscale model parallel.
    """
    import torch.distributed as dist
    import fairscale.nn.model_parallel.initialize as fs_init
    
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    jobid         = int(os.environ["SLURM_JOBID"])
    gpus_per_node = torch.cuda.device_count()
    print('jobid ', jobid)
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
          f" {gpus_per_node} allocated GPUs per node. ", flush=True)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"rank: {rank}, local_rank: {local_rank}")
    
    # Initialize fairscale model parallel for LLaMA
    if not fs_init.model_parallel_is_initialized():
        fs_init.initialize_model_parallel(1)  # 1 = use 1 GPU for model parallel
        if rank == 0: print("Fairscale model parallel initialized", flush=True)
    
    return local_rank, world_size, gpus_per_node
    

# Example usage
if __name__ == "__main__":
    print("Example usage of MultimodalStellarModel:")

    device, world_size, apus_per_node = setup()
    
    # Load pre-trained models
    llm_model = _load_llm_model()  # Or your preferred LLM
    spectral_model = _load_spectra_model()  # Your spectral model

    # 2. Configure alignment
    config = AlignmentConfig(
        text_embedding_dim=384,  # MiniLM embedding dim
        spectral_embedding_dim=512,  # Your spectral model output dim
        projection_dim=256,
        text_projection_hidden_dims=[512, 256],
        temperature=0.07,
        learnable_temperature=True
    )

    # 3. Create multimodal model
    model = MultimodalStellarModel(
        llm_model=llm_model,
        spectral_model=spectral_model,
        alignment_config=config,
        freeze_llm=True,
        freeze_spectral=True
    )

    print(model)