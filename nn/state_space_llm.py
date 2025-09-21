import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Dict, Union
from nn.llm import apply_rotary_emb, repeat_kv, LatentFeatureEncoder 

class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices - initialize on correct device/dtype
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features, device=device, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Ensure LoRA params match input dtype/device (handles AMP/fp16)
        if self.lora_A.dtype != x.dtype:
            self.lora_A.data = self.lora_A.data.to(dtype=x.dtype)
            self.lora_B.data = self.lora_B.data.to(dtype=x.dtype)
        if self.lora_A.device != x.device:
            self.lora_A.data = self.lora_A.data.to(device=x.device)
            self.lora_B.data = self.lora_B.data.to(device=x.device)

        # x shape: (..., in_features)
        # LoRA forward: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return self.dropout(lora_out)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation - supports both regular Linear and FairScale parallel layers"""

    def __init__(self, original_layer, rank: int = 16, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer

        # Get device and dtype from original layer
        p = next(original_layer.parameters())
        device = p.device
        dtype = p.dtype

        # Handle different layer types
        if hasattr(original_layer, 'in_features') and hasattr(original_layer, 'out_features'):
            # Regular nn.Linear
            in_features = original_layer.in_features
            out_features = original_layer.out_features
        elif hasattr(original_layer, 'input_size') and hasattr(original_layer, 'output_size'):
            # FairScale parallel layers
            in_features = original_layer.input_size
            out_features = original_layer.output_size
        else:
            # Fallback - inspect weight shape
            weight = original_layer.weight
            out_features, in_features = weight.shape

        self.lora = LoRALayer(
            in_features,
            out_features,
            rank,
            alpha,
            dropout,
            device=device,
            dtype=dtype,
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
    
    # Get supported linear layer types
    try:
        from fairscale.nn.model_parallel.layers import RowParallelLinear, ColumnParallelLinear
        linear_types = (nn.Linear, RowParallelLinear, ColumnParallelLinear)
        print("Using FairScale parallel layers support")
    except ImportError:
        linear_types = (nn.Linear,)
        print("FairScale not available, using only torch.nn.Linear")
    
    lora_modules = {}

    def replace_with_lora(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this is a target module - EXACT MATCH
            if full_name in target_modules and isinstance(child_module, linear_types):
                print(f"Applying LoRA to: {full_name} ({type(child_module).__name__})")
                lora_layer = LoRALinear(child_module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)
                lora_modules[full_name] = lora_layer
            else:
                # Recursively apply to children
                replace_with_lora(child_module, full_name)

    replace_with_lora(model)
    return lora_modules



class EvolutionaryFeatureEncoder(nn.Module):
    """
    Enhanced encoder for evolutionary stellar features that can handle:
    1. Single-stage features
    2. Multi-stage evolutionary sequences
    3. Stage-specific embeddings
    """

    def init(self, latent_dim: int, embedding_dim: int, hidden_dim: int = 512,
                 max_stages: int = 10):
        super().init()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.max_stages = max_stages

        # Main feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # Stage-specific embeddings (learnable embeddings for each evolutionary stage)
        self.stage_embeddings = nn.Embedding(max_stages, embedding_dim)

        # Optional: combination layer to merge feature and stage embeddings
        self.combination_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU()
        )

    def forward(self, features: torch.Tensor, stage_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (batch_size, latent_dim) or (batch_size, n_stages, latent_dim)
            stage_indices: (batch_size,) stage indices for single features, or
                          (batch_size, n_stages) for multi-stage features
        Returns:
            embeddings: (batch_size, embedding_dim) or (batch_size, n_stages, embedding_dim)
        """
        # Handle different input shapes
        if len(features.shape) == 2:
            # Single-stage features: (batch_size, latent_dim)
            feature_embeds = self.feature_encoder(features)  # (batch_size, embedding_dim)

            if stage_indices is not None:
                # Add stage-specific information
                stage_embeds = self.stage_embeddings(stage_indices)  # (batch_size, embedding_dim)
                combined = torch.cat([feature_embeds, stage_embeds], dim=-1)
                return self.combination_layer(combined)
            else:
                return feature_embeds

        elif len(features.shape) == 3:
            # Multi-stage features: (batch_size, n_stages, latent_dim)
            batch_size, n_stages, latent_dim = features.shape

            # Reshape for batch processing
            features_flat = features.view(-1, latent_dim)  # (batch_size * n_stages, latent_dim)
            feature_embeds = self.feature_encoder(features_flat)  # (batch_size * n_stages, embedding_dim)
            feature_embeds = feature_embeds.view(batch_size, n_stages, -1)  # (batch_size, n_stages, embedding_dim)

            if stage_indices is not None:
                # stage_indices should be (batch_size, n_stages)
                stage_embeds = self.stage_embeddings(stage_indices)  # (batch_size, n_stages, embedding_dim)

                # Combine feature and stage embeddings
                combined = torch.cat([feature_embeds, stage_embeds],
                                     dim=-1)  # (batch_size, n_stages, embedding_dim * 2)
                combined_flat = combined.view(-1, self.embedding_dim * 2)
                result = self.combination_layer(combined_flat)  # (batch_size * n_stages, embedding_dim)
                return result.view(batch_size, n_stages, -1)  # (batch_size, n_stages, embedding_dim)
            else:
                return feature_embeds
        else:
            raise ValueError(f"Unsupported features shape: {features.shape}")

class EvolutionaryFeatureAutoencoder(nn.Module):
    """
    Enhanced encoder-decoder for evolutionary stellar features that can handle:
    1. Single-stage features
    2. Multi-stage evolutionary sequences
    3. Stage-specific embeddings
    4. Reconstruction of original features
    """

    def __init__(self, latent_dim: int, embedding_dim: int, hidden_dim: int = 512,
                 max_stages: int = 10, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_stages = max_stages

        # ============ ENCODER ============
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # Stage-specific embeddings (learnable embeddings for each evolutionary stage)
        self.stage_embeddings = nn.Embedding(max_stages, embedding_dim)

        # Combination layer to merge feature and stage embeddings
        self.encoder_combination = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU()
        )

        # ============ DECODER ============
        # Stage-aware decoder that can reconstruct from encoded features
        self.decoder_preparation = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),  # Takes encoded + stage info
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Optional: Residual connection for better reconstruction
        self.use_residual = True
        if self.use_residual:
            self.residual_gate = nn.Sequential(
                nn.Linear(embedding_dim, latent_dim),
                nn.Sigmoid()
            )

    def encode(self, features: torch.Tensor, stage_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode features into latent space

        Args:
            features: (batch_size, latent_dim) or (batch_size, n_stages, latent_dim)
            stage_indices: (batch_size,) or (batch_size, n_stages)
        Returns:
            encoded_features: Same shape as input but with embedding_dim as last dimension
        """
        if len(features.shape) == 2:
            # Single-stage features: (batch_size, latent_dim)
            encoded = self.encoder(features)  # (batch_size, embedding_dim)

            if stage_indices is not None:
                stage_embeds = self.stage_embeddings(stage_indices)  # (batch_size, embedding_dim)
                combined = torch.cat([encoded, stage_embeds], dim=-1)
                return self.encoder_combination(combined)
            return encoded

        elif len(features.shape) == 3:
            # Multi-stage features: (batch_size, n_stages, latent_dim)
            batch_size, n_stages, latent_dim = features.shape

            # Reshape for batch processing
            features_flat = features.view(-1, latent_dim)
            encoded_flat = self.encoder(features_flat)  # (batch_size * n_stages, embedding_dim)
            encoded = encoded_flat.view(batch_size, n_stages, self.embedding_dim)

            if stage_indices is not None:
                stage_embeds = self.stage_embeddings(stage_indices)  # (batch_size, n_stages, embedding_dim)
                combined = torch.cat([encoded, stage_embeds], dim=-1)
                combined_flat = combined.view(-1, self.embedding_dim * 2)
                result = self.encoder_combination(combined_flat)
                return result.view(batch_size, n_stages, self.embedding_dim)
            return encoded
        else:
            raise ValueError(f"Unsupported features shape: {features.shape}")

    def decode(self, encoded_features: torch.Tensor, stage_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode features from latent space back to original space

        Args:
            encoded_features: (batch_size, embedding_dim) or (batch_size, n_stages, embedding_dim)
            stage_indices: (batch_size,) or (batch_size, n_stages)
        Returns:
            reconstructed_features: Same shape as original input features
        """
        if len(encoded_features.shape) == 2:
            # Single-stage decoding
            if stage_indices is not None:
                stage_embeds = self.stage_embeddings(stage_indices)
                decoder_input = torch.cat([encoded_features, stage_embeds], dim=-1)
            else:
                # Use zero stage embedding if not provided
                zero_stage = torch.zeros_like(encoded_features)
                decoder_input = torch.cat([encoded_features, zero_stage], dim=-1)

            hidden = self.decoder_preparation(decoder_input)
            reconstructed = self.decoder(hidden)

            # Optional residual connection
            if self.use_residual:
                gate = self.residual_gate(encoded_features)
                reconstructed = reconstructed * gate + reconstructed * (1 - gate)

            return reconstructed

        elif len(encoded_features.shape) == 3:
            # Multi-stage decoding
            batch_size, n_stages, embedding_dim = encoded_features.shape

            if stage_indices is not None:
                stage_embeds = self.stage_embeddings(stage_indices)
            else:
                # Use sequential stage indices if not provided
                stage_indices = torch.arange(n_stages, device=encoded_features.device).unsqueeze(0).expand(batch_size,
                                                                                                           -1)
                stage_embeds = self.stage_embeddings(stage_indices)

            # Combine encoded features with stage embeddings
            decoder_input = torch.cat([encoded_features, stage_embeds], dim=-1)
            decoder_input_flat = decoder_input.view(-1, self.embedding_dim * 2)

            # Decode
            hidden = self.decoder_preparation(decoder_input_flat)
            reconstructed_flat = self.decoder(hidden)
            reconstructed = reconstructed_flat.view(batch_size, n_stages, self.latent_dim)

            return reconstructed
        else:
            raise ValueError(f"Unsupported encoded_features shape: {encoded_features.shape}")

    def forward(self, features: torch.Tensor, stage_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode then decode

        Args:
            features: (batch_size, latent_dim) or (batch_size, n_stages, latent_dim)
            stage_indices: (batch_size,) or (batch_size, n_stages)
        Returns:
            Dict containing:
                - 'encoded': encoded features
                - 'reconstructed': reconstructed features
        """
        encoded = self.encode(features, stage_indices)
        reconstructed = self.decode(encoded, stage_indices)

        return {
            'encoded': encoded,
            'reconstructed': reconstructed
        }

    def get_reconstruction_loss(self, features: torch.Tensor, stage_indices: Optional[torch.Tensor] = None,
                                loss_fn: str = 'mse') -> torch.Tensor:
        """
        Compute reconstruction loss

        Args:
            features: Original features
            stage_indices: Stage indices
            loss_fn: 'mse' or 'l1'
        Returns:
            reconstruction_loss: Scalar loss
        """
        output = self.forward(features, stage_indices)
        reconstructed = output['reconstructed']

        if loss_fn == 'mse':
            return nn.functional.mse_loss(reconstructed, features)
        elif loss_fn == 'l1':
            return nn.functional.l1_loss(reconstructed, features)
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")

    def encode_only(self, features: torch.Tensor, stage_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convenience method for encoding only"""
        return self.encode(features, stage_indices)


class StellarEvolutionStateSpace(nn.Module):
    """
    State-space model for predicting stellar evolution sequences
    Predicts next evolutionary state given previous states and current stellar parameters
    """

    def __init__(self,
                 feature_dim: int,
                 hidden_dim: int = 256,
                 n_layers: int = 4,
                 model_type: str = "lstm",  # "lstm", "transformer"
                 max_age: float = 15.0,
                 age_encoding_dim: int = 64):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type
        self.max_age = max_age
        self.age_encoding_dim = age_encoding_dim

        # Parameter encoders
        self.age_encoder = nn.Sequential(
            nn.Linear(1, age_encoding_dim // 2),
            nn.SiLU(),
            nn.Linear(age_encoding_dim // 2, age_encoding_dim)
        )

        self.mass_encoder = nn.Sequential(
            nn.Linear(1, age_encoding_dim // 2),
            nn.SiLU(),
            nn.Linear(age_encoding_dim // 2, age_encoding_dim)
        )

        self.metallicity_encoder = nn.Sequential(
            nn.Linear(1, age_encoding_dim // 4),
            nn.SiLU(),
            nn.Linear(age_encoding_dim // 4, age_encoding_dim // 2)
        )

        # Initial stage generator
        param_encoding_dim = age_encoding_dim + age_encoding_dim + age_encoding_dim // 2
        self.initial_stage_generator = nn.Sequential(
            nn.Linear(param_encoding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )

        # Input projection: features + parameters
        input_dim = feature_dim + param_encoding_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Core sequential model
        if model_type == "lstm":
            self.sequential_model = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=0.1 if n_layers > 1 else 0.0
            )
        elif model_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.sequential_model = nn.TransformerEncoder(encoder_layer, n_layers)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Output prediction head
        self.evolution_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, feature_dim)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def encode_stellar_parameters(self, ages: torch.Tensor, masses: torch.Tensor,
                                  metallicities: torch.Tensor) -> torch.Tensor:
        """
        Encode stellar parameters into embeddings

        Args:
            ages: (batch_size, n_stages) ages in Gyr
            masses: (batch_size,) or (batch_size, n_stages) stellar masses in M_sun
            metallicities: (batch_size,) or (batch_size, n_stages) [Fe/H]

        Returns:
            param_encodings: (batch_size, n_stages, encoding_dim)
        """
        batch_size, n_stages = ages.shape

        # Normalize age to [0, 1] range
        normalized_ages = ages / self.max_age
        age_encodings = self.age_encoder(normalized_ages.unsqueeze(-1))  # (B, T, age_dim)

        # Handle mass broadcasting
        if len(masses.shape) == 1:
            masses = masses.unsqueeze(-1).expand(-1, n_stages)  # (B, T)
        mass_encodings = self.mass_encoder(masses.unsqueeze(-1))  # (B, T, mass_dim)

        # Handle metallicity broadcasting
        if len(metallicities.shape) == 1:
            metallicities = metallicities.unsqueeze(-1).expand(-1, n_stages)  # (B, T)
        met_encodings = self.metallicity_encoder(metallicities.unsqueeze(-1))  # (B, T, met_dim)

        # Combine all parameter encodings
        param_encodings = torch.cat([age_encodings, mass_encodings, met_encodings], dim=-1)

        return param_encodings

    def create_autoregressive_input_sequence(self,
                                             enhanced_features: torch.Tensor,
                                             zero_features: torch.Tensor,
                                             ages: torch.Tensor,
                                             masses: torch.Tensor,
                                             metallicities: torch.Tensor,
                                             stage_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Create input sequence for autoregressive prediction without loops

        Key insight: For predicting stage t+1, we need:
        - Features from stage t (current)
        - Parameters for stage t+1 (next)

        So we create sequences where:
        - Input features: [stage_0, stage_1, ..., stage_{T-2}]
        - Input parameters: [params_1, params_2, ..., params_{T-1}]
        - Target features: [stage_1, stage_2, ..., stage_{T-1}]
        """
        batch_size, n_stages, feature_dim = enhanced_features.shape
        device = enhanced_features.device

        enhanced_features = torch.cat([zero_features, enhanced_features], dim=1)

        # Current features: stages 0 to T-2 (what we predict FROM)
        current_features = enhanced_features[:, :-1, :]  # (B, T-1, D)

        # Target features: stages 1 to T-1 (what we're trying to predict)
        target_features = enhanced_features[:, 1:, :]  # (B, T-1, D)

        # Encode next stage parameters
        next_param_encodings = self.encode_stellar_parameters(
            ages, masses, metallicities
        )  # (B, T-1, param_dim)

        # Combine current features + next parameters
        combined_input = torch.cat([current_features, next_param_encodings], dim=-1)

        # Project to hidden dimension
        input_sequence = self.input_projection(combined_input)  # (B, T-1, hidden_dim)
        input_sequence = self.layer_norm(input_sequence)


        return {
            'input_sequence': input_sequence,
            'prediction_mask': stage_mask,
            'target_features': target_features
        }

    def forward(self,
                enhanced_features: torch.Tensor,
                zero_features: torch.tensor,
                ages: torch.Tensor,
                masses: torch.Tensor,
                metallicities: torch.Tensor,
                stage_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Vectorized forward pass - no loops!

        Args:
            enhanced_features: (batch_size, n_stages, feature_dim) LLM-enhanced features
            ages: (batch_size, n_stages) ages for each stage
            masses: (batch_size,) or (batch_size, n_stages) stellar masses
            metallicities: (batch_size,) or (batch_size, n_stages) metallicities
            stage_mask: (batch_size, n_stages) mask for real vs padded stages

        Returns:
            Dictionary with predictions
        """
        batch_size, n_stages, feature_dim = enhanced_features.shape
        device = enhanced_features.device

        # Create autoregressive input sequence
        sequence_data = self.create_autoregressive_input_sequence(
            enhanced_features, zero_features, ages, masses, metallicities, stage_mask
        )

        input_sequence = sequence_data['input_sequence']  # (B, T-1, hidden_dim)
        prediction_mask = sequence_data['prediction_mask']  # (B, T-1)
        target_features = sequence_data['target_features']  # (B, T-1, feature_dim)

        # Apply mask to input sequence
        input_sequence = input_sequence * prediction_mask.unsqueeze(-1).float()

        # Process through sequential model
        if self.model_type == "lstm":
            # LSTM processes the entire sequence at once
            # Pack padded sequence for efficiency
            lengths = prediction_mask.sum(dim=1).cpu()

            if lengths.min() > 0:  # Only pack if all sequences have length > 0
                packed_input = nn.utils.rnn.pack_padded_sequence(
                    input_sequence, lengths, batch_first=True, enforce_sorted=False
                )
                packed_output, _ = self.sequential_model(packed_input)
                h, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            else:
                h, _ = self.sequential_model(input_sequence)

        elif self.model_type == "transformer":
            # Create causal mask for autoregressive prediction
            seq_len = input_sequence.shape[1]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()

            # Create key padding mask
            key_padding_mask = ~prediction_mask  # True for padded positions

            h = self.sequential_model(
                input_sequence,
                mask=causal_mask,
                src_key_padding_mask=key_padding_mask
            )

        # Generate predictions for each position
        evolution_predictions = self.evolution_predictor(h)  # (B, T-1, feature_dim)

        # Apply final mask
        evolution_predictions = evolution_predictions * prediction_mask.unsqueeze(-1).float()

        return {
            'evolution_predictions': evolution_predictions,
            'hidden_states': h,
            'target_features': target_features,
            'prediction_mask': prediction_mask
        }


class PhysicsInformedEvolutionaryLlama(nn.Module):
    """
    Enhanced LLaMA model with physics-informed evolutionary state-space modeling
    """

    def __init__(self, base_model, latent_dim: int,
                 special_tokens: dict = None, max_stages: int = 4,
                 evolution_model_type: str = "lstm"):
        super().__init__()

        self.base_model = base_model
        self.embedding_dim = base_model.params.dim
        self.special_tokens = special_tokens
        self.max_stages = max_stages

        # Feature encoder for LLM integration
        self.feature_encoder = EvolutionaryFeatureAutoencoder(
            latent_dim=latent_dim,
            embedding_dim=self.embedding_dim,
            hidden_dim=512,
            max_stages=max_stages
        )

        # Physics-informed evolutionary state-space model
        self.evolution_model = StellarEvolutionStateSpace(
            feature_dim=latent_dim,
            hidden_dim=256,
            n_layers=4,
            model_type=evolution_model_type,
            max_age=15.0,
            age_encoding_dim=64
        )

    def forward(self,
                input_ids: torch.Tensor,
                features: torch.Tensor,
                stage_mask: torch.Tensor,
                n_stages: torch.Tensor,
                ages: torch.Tensor,
                masses: torch.Tensor,
                metallicities: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,  # NEW: Accept attention mask
                start_pos: int = 0,
                use_enhanced_features: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with both LLM and evolution modeling
        Now properly uses attention masks

        Args:
            input_ids: (batch_size, seq_len) token IDs
            features: (batch_size, max_stages, latent_dim) padded features
            stage_mask: (batch_size, max_stages) mask for real vs padded stages
            n_stages: (batch_size,) number of real stages per example
            ages: (batch_size, max_stages) ages for each stage
            masses: (batch_size,) stellar masses
            metallicities: (batch_size,) metallicities
            attention_mask: (batch_size, seq_len) attention mask for input tokens
            use_enhanced_features: Whether to use LLM-enhanced features for physics model
        """

        # 1. LLM forward pass (for text generation) with attention mask
        llm_outputs = self._forward_llm(input_ids, features, stage_mask, n_stages, attention_mask)

        zero_features = torch.zeros(llm_outputs['h'].shape[0], 1,
                                   llm_outputs['h'].shape[-1], device=features.device)
        
        # 2. Extract enhanced features directly from LLM hidden states
        enhanced_features = self._extract_enhanced_features(
            features,
            llm_outputs['h'],
            llm_outputs['special_positions'],
            llm_outputs['feature_embeddings'],
            llm_outputs['raw_features'],
            stage_mask
        )

        # 3. Evolution model forward pass with causal masking
        evolution_outputs = self.evolution_model(
            enhanced_features, zero_features, ages, masses, metallicities, stage_mask
        )

        features_decoded = self.feature_encoder.decode(enhanced_features, stage_mask.int())
        features_decoded = features_decoded * stage_mask.unsqueeze(-1).float()

        return {
            # LLM outputs
            'logits': llm_outputs['logits'],
            'hidden_states': llm_outputs['h'],
            'special_positions': llm_outputs['special_positions'],
            'feature_embeddings': llm_outputs['feature_embeddings'],
            'stage_mask': stage_mask,

            # Features
            'enhanced_features': enhanced_features,

            # Evolution model outputs
            'evolution_predictions': evolution_outputs['evolution_predictions'],
            'evolution_targets': evolution_outputs['target_features'],
            'features_decoded': features_decoded,
            'features': features
        }

    def _extract_enhanced_features(self,
                                   raw_features: torch.Tensor,
                                   llm_hidden_states: torch.Tensor,
                                   special_positions: Dict[Tuple[int, int], int],
                                   feature_embeddings: Dict[Tuple[int, int, int], torch.Tensor],
                                   features_dict: Dict[Tuple[int, int, int], torch.Tensor],
                                   stage_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract enhanced features directly from LLM hidden states at special token positions
        Uses the saved feature_embeddings instead of re-encoding

        Args:
            raw_features: (batch_size, max_stages, feature_dim) original features
            llm_hidden_states: (batch_size, seq_len, hidden_dim) LLM hidden states
            special_positions: Dict mapping (batch_idx, token_pos) -> stage_idx
            feature_embeddings: Dict mapping (batch_idx, token_pos, stage_idx) -> original feature embedding
            stage_mask: (batch_size, max_stages) mask for real stages

        Returns:
            enhanced_features: (batch_size, max_stages, feature_dim) enhanced features
        """
        # true_features = raw_features.clone()
        enhanced_features = torch.zeros_like(raw_features)

        # Extract enhanced features for each stage
        for (batch_idx, token_pos), stage_idx in special_positions.items():
            if stage_idx < raw_features.shape[1] and stage_mask[batch_idx, stage_idx]:
                # Extract LLM-processed feature directly (already processed by transformer layers)
                llm_feature = llm_hidden_states[batch_idx, token_pos, :]  # (hidden_dim,)

                projected_feature = feature_embeddings[(batch_idx, token_pos, stage_idx)]  # (feature_dim,)
                # true_feature = features_dict[(batch_idx, token_pos, stage_idx)]

                enhanced_features[batch_idx, stage_idx, :] = llm_feature + projected_feature
                # enhanced_features[batch_idx, stage_idx, :] = projected_feature
                # true_features[batch_idx, stage_idx, :] = true_feature

        return enhanced_features

    def _forward_llm(self, input_ids: torch.Tensor, features: torch.Tensor,
                     stage_mask: torch.Tensor, n_stages: torch.Tensor, 
                     attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through LLM component with attention mask support"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get token embeddings
        token_embeddings = self.base_model.tok_embeddings(input_ids)

        # Find and replace special token embeddings
        special_positions = {}  # Maps (batch_idx, token_pos) -> stage_idx
        feature_embeddings = {}  # Maps (batch_idx, token_pos, stage_idx) -> feature_embedding
        raw_features = {}

        for batch_idx in range(batch_size):
            for token_pos in range(seq_len):
                token_id = input_ids[batch_idx, token_pos].item()
                for token_name, special_id in self.special_tokens.items():
                    if token_id == special_id:
                        stage_idx = int(token_name.split('_')[-1])
                        if stage_idx < n_stages[batch_idx].item():  # Only process valid stages
                            special_positions[(batch_idx, token_pos)] = stage_idx

                            # Get stage features
                            stage_features = features[batch_idx, stage_idx]
                            stage_tensor = torch.tensor([stage_idx], device=device)

                            # Encode features to embedding space
                            feature_embedding = self.feature_encoder.encode(
                                stage_features.unsqueeze(0),
                                stage_tensor
                            ).squeeze(0)

                            # Replace token embedding
                            token_embeddings[batch_idx, token_pos] = feature_embedding

                            # Store for analysis
                            feature_embeddings[(batch_idx, token_pos, stage_idx)] = feature_embedding
                            raw_features[(batch_idx, token_pos, stage_idx)] = stage_features
                        break

        # Forward through transformer WITH attention mask
        out, h = self._simple_forward(token_embeddings, attention_mask)

        return {
            "logits": out,
            "h": h,
            "special_positions": special_positions,
            'feature_embeddings': feature_embeddings,
            'raw_features': raw_features
        }

    def _simple_forward(self, embeddings: torch.Tensor, 
                        attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple transformer forward pass with attention mask support"""
        batch_size, seq_len, embed_dim = embeddings.shape
        device = embeddings.device

        h = embeddings

        # Create RoPE frequencies
        head_dim = self.base_model.params.dim // self.base_model.params.n_heads
        freqs = 1.0 / (self.base_model.params.rope_theta ** (
                torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        # Create causal mask for language modeling
        causal_mask = None
        if seq_len > 1:
            causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=h.dtype)
            causal_mask = torch.triu(causal_mask, diagonal=1)

        # Convert attention mask to the format needed for transformer
        # attention_mask: (batch_size, seq_len) where 1 = attend, 0 = don't attend
        # We need to convert to additive mask: 0 = attend, -inf = don't attend
        transformer_attention_mask = None
        if attention_mask is not None:
            # Convert from (batch_size, seq_len) to (batch_size, 1, 1, seq_len) for broadcasting
            transformer_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            transformer_attention_mask = (1.0 - transformer_attention_mask) * -10000.0  # Convert to additive mask
            
            # Expand to (batch_size, num_heads, seq_len, seq_len) for each attention head
            num_heads = self.base_model.params.n_heads
            transformer_attention_mask = transformer_attention_mask.expand(batch_size, num_heads, seq_len, seq_len)

        # Pass through transformer layers
        for layer in self.base_model.layers:
            h_norm = layer.attention_norm(h)
            attn_out = self._simple_attention(h_norm, freqs_cis, causal_mask, layer.attention, transformer_attention_mask)
            h = h + attn_out

            ff_norm = layer.ffn_norm(h)
            ff_out = layer.feed_forward(ff_norm)
            h = h + ff_out

        h = self.base_model.norm(h)
        output = self.base_model.output(h).float()

        return output, h

    def _simple_attention(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                          causal_mask: Optional[torch.Tensor], attention_layer,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Simplified attention computation with proper mask support"""
        bsz, seqlen, _ = x.shape

        xq = attention_layer.wq(x)
        xk = attention_layer.wk(x)
        xv = attention_layer.wv(x)

        xq = xq.view(bsz, seqlen, attention_layer.n_local_heads, attention_layer.head_dim)
        xk = xk.view(bsz, seqlen, attention_layer.n_local_kv_heads, attention_layer.head_dim)
        xv = xv.view(bsz, seqlen, attention_layer.n_local_kv_heads, attention_layer.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        keys = repeat_kv(xk, attention_layer.n_rep)
        values = repeat_kv(xv, attention_layer.n_rep)

        xq = xq.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(attention_layer.head_dim)
        # scores shape: (batch_size, n_heads, seq_len, seq_len)

        # Apply causal mask (for autoregressive generation)
        if causal_mask is not None:
            scores = scores + causal_mask

        # Apply attention mask (to ignore padding tokens)
        if attention_mask is not None:
            scores = scores + attention_mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return attention_layer.wo(output)


def compute_evolution_loss(outputs: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor],
                           stage_mask: torch.Tensor,
                           loss_weights: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
    """
    Compute physics-informed loss for evolutionary predictions

    Args:
        outputs: Model outputs with evolution predictions
        targets: Target values for stellar parameters
        stage_mask: (batch_size, max_stages) mask for real stages
        loss_weights: Weights for different loss components
    """
    if loss_weights is None:
        loss_weights = {
            'evolution': 1.0,
            'temperature': 1.0,
            'gravity': 1.0,
            'luminosity': 1.0
        }

    device = stage_mask.device
    losses = {}

    # Create causal mask for evolution predictions (each stage predicts the next)
    batch_size, max_stages = stage_mask.shape

    # For evolution prediction, we predict stage t+1 from stages 0:t
    if max_stages > 1:
        evolution_mask = stage_mask[:, 1:]  # Exclude first stage (nothing to predict from)

        if 'evolution_predictions' in outputs and 'features' in targets:
            # Predict next evolutionary state
            pred_evolution = outputs['evolution_predictions'][:, :-1, :]  # (B, T-1, D)
            target_evolution = targets['features'][:, 1:, :]  # (B, T-1, D)

            evolution_loss = F.mse_loss(
                pred_evolution * evolution_mask.unsqueeze(-1),
                target_evolution * evolution_mask.unsqueeze(-1),
                reduction='sum'
            ) / (evolution_mask.sum() + 1e-8)

            losses['evolution'] = evolution_loss * loss_weights['evolution']

        # Physical parameter predictions
        for param_name, pred_key in [
            ('temperature', 'temperature_predictions'),
            ('gravity', 'gravity_predictions'),
            ('luminosity', 'luminosity_predictions')
        ]:
            if pred_key in outputs and param_name in targets:
                pred_param = outputs[pred_key][:, :-1]  # (B, T-1)
                target_param = targets[param_name][:, 1:]  # (B, T-1)

                param_loss = F.mse_loss(
                    pred_param * evolution_mask,
                    target_param * evolution_mask,
                    reduction='sum'
                ) / (evolution_mask.sum() + 1e-8)

                losses[param_name] = param_loss * loss_weights[param_name]

    # Total physics loss
    physics_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=device)
    losses['total_physics'] = physics_loss

    return losses


def create_physics_informed_llama(base_model, latent_dim: int, tokenizer,
                                  special_token_pattern: str = "STAR_DATA",
                                  max_stages: int = 10,
                                  evolution_model_type: str = "mamba"):
    """
    Create physics-informed evolutionary LLaMA model
    """
    model = PhysicsInformedEvolutionaryLlama(
        base_model, latent_dim, special_token_pattern, max_stages, evolution_model_type
    )

    model.register_special_tokens(tokenizer)

    return model


# Training function
def train_step_physics_informed(model: PhysicsInformedEvolutionaryLlama,
                                batch: Dict, optimizer: torch.optim.Optimizer,
                                device: str = 'cuda',
                                alpha_llm: float = 1.0,
                                alpha_physics: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Training step combining LLM and physics losses

    Args:
        alpha_llm: Weight for language modeling loss
        alpha_physics: Weight for physics evolution loss
    """

    # Extract batch data
    input_ids = batch['input_ids'].to(device)
    target_ids = batch.get('target_ids', input_ids[:, 1:]).to(device)  # Shifted for LM
    features = batch['features'].to(device)
    stage_mask = batch['stage_mask'].to(device)
    n_stages = batch['n_stages'].to(device)

    # Extract stellar parameters from metadata if available
    metadata = batch.get('metadata', [{}] * len(input_ids))
    ages = torch.zeros(len(input_ids), features.shape[1], device=device)
    masses = torch.zeros(len(input_ids), device=device)
    metallicities = torch.zeros(len(input_ids), device=device)

    for i, meta in enumerate(metadata):
        if 'evolutionary_stages' in meta:
            stages = meta['evolutionary_stages']
            for j, stage in enumerate(stages):
                if j < ages.shape[1]:
                    ages[i, j] = stage.get('age', 0.0)

        masses[i] = meta.get('initial_mass', meta.get('mass', 1.0))
        metallicities[i] = meta.get('metallicity', meta.get('feh', 0.0))

    # Forward pass
    outputs = model(input_ids, features, stage_mask, n_stages,
                    ages, masses, metallicities)

    # Language modeling loss
    if alpha_llm > 0:
        logits = outputs['logits'][:, :-1, :]  # Exclude last token
        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction='mean'
        )
    else:
        lm_loss = torch.tensor(0.0, device=device)

    # Physics evolution loss
    if alpha_physics > 0:
        # Create targets from actual data
        targets = {
            'features': features,
            # You would extract these from your stellar metadata
            # 'temperature': ...,
            # 'gravity': ...,
            # 'luminosity': ...
        }

        physics_losses = compute_evolution_loss(outputs, targets, stage_mask)
        physics_loss = physics_losses['total_physics']
    else:
        physics_loss = torch.tensor(0.0, device=device)
        physics_losses = {}

    # Combined loss
    total_loss = alpha_llm * lm_loss + alpha_physics * physics_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        'total_loss': total_loss,
        'lm_loss': lm_loss,
    }
