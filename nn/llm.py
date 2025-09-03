import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

TOKENIZER_PATH = "/data/.llama/checkpoints/Llama3.1-8B/tokenizer.model"
MODEL_PATH = "/data/.llama/checkpoints/Llama3.1-8B"

def memory_profile(func):
    """Decorator to profile memory usage of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
            device = torch.cuda.current_device()
            before_allocated = torch.cuda.memory_allocated(device) / 1024**3
            before_reserved = torch.cuda.memory_reserved(device) / 1024**3
            
            print(f"\n{'='*50}")
            print(f"BEFORE {func.__name__}:")
            print(f"  Allocated: {before_allocated:.2f} GB")
            print(f"  Reserved: {before_reserved:.2f} GB")
            
            try:
                result = func(*args, **kwargs)
                
                after_allocated = torch.cuda.memory_allocated(device) / 1024**3
                after_reserved = torch.cuda.memory_reserved(device) / 1024**3
                
                print(f"AFTER {func.__name__}:")
                print(f"  Allocated: {after_allocated:.2f} GB (+{after_allocated-before_allocated:.2f} GB)")
                print(f"  Reserved: {after_reserved:.2f} GB (+{after_reserved-before_reserved:.2f} GB)")
                print(f"{'='*50}\n")
                
                return result
                
            except Exception as e:
                current_allocated = torch.cuda.memory_allocated(device) / 1024**3
                current_reserved = torch.cuda.memory_reserved(device) / 1024**3
                
                print(f"ERROR in {func.__name__}:")
                print(f"  Allocated: {current_allocated:.2f} GB (+{current_allocated-before_allocated:.2f} GB)")
                print(f"  Reserved: {current_reserved:.2f} GB (+{current_reserved-before_reserved:.2f} GB)")
                print(f"  Error: {str(e)}")
                print(f"{'='*50}\n")
                raise
        else:
            return func(*args, **kwargs)
    return wrapper

def print_tensor_memory(tensor, name="tensor"):
    """Print memory usage of a tensor"""
    if torch.cuda.is_available() and tensor.is_cuda:
        size_mb = tensor.element_size() * tensor.numel() / 1024**2
        print(f"  {name}: {tensor.shape} -> {size_mb:.2f} MB")
        if tensor.requires_grad:
            print(f"    (requires_grad=True - will store gradients!)")

def trace_gradient_computation(model):
    """Add hooks to trace which parts of the model are computing gradients"""
    def forward_hook(name):
        def hook(module, input, output):
            if hasattr(output, 'requires_grad') and output.requires_grad:
                print(f"  FORWARD {name}: output requires_grad={output.requires_grad}")
                if hasattr(output, 'shape'):
                    print_tensor_memory(output, f"{name}_output")
        return hook
    
    def backward_hook(name):
        def hook(module, grad_input, grad_output):
            print(f"  BACKWARD {name}: computing gradients")
            if grad_output[0] is not None:
                print_tensor_memory(grad_output[0], f"{name}_grad")
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hooks.append(module.register_forward_hook(forward_hook(name)))
            hooks.append(module.register_backward_hook(backward_hook(name)))
    
    return hooks

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply rotary positional embedding to the query and key tensors."""

    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value tensors n_rep times."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class LatentFeatureEncoder(nn.Module):
    """MLP encoder to project latent features to token embedding space"""

    def __init__(self, latent_dim: int, embedding_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, latent_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_features: (batch_size, latent_dim) or (latent_dim,)
        Returns:
            embeddings: (batch_size, embedding_dim) or (embedding_dim,)
        """
        return self.encoder(latent_features)


class MultimodalLlamaModel(nn.Module):
    """LLaMA model with multimodal capabilities for latent feature integration"""

    def __init__(self, base_model, fm_model, latent_dim, hidden_dim, num_spectral_features: int = 1):
        super().__init__()
        self.base_model = base_model
        self.fm_model = fm_model
        self.embedding_dim = base_model.params.dim
        print("Embedding dimension:", self.embedding_dim)
        self.num_spectral_features = num_spectral_features

        # Latent feature encoder - projects features to embedding space
        self.latent_encoder = LatentFeatureEncoder(
            latent_dim=latent_dim,
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim
        )


    def forward(self,
                input_ids: torch.Tensor,
                input_spectra: torch.Tensor,
                special_token_positions: torch.Tensor,
                start_pos: int = 0) -> Dict[str, torch.Tensor]:
        """
        Forward pass with latent feature integration at embedding level
        """
        self.fm_model.eval()  # Ensure fm_model is in eval mode
        batch_size, seq_len = input_ids.shape
        with torch.no_grad():
            _, _, latent_features = self.fm_model(input_spectra)
        if self.num_spectral_features <= 1:
            latent_features = latent_features.sum(dim=1)  # (batch_size, fm_features_dim)     

        return self._forward_training(input_ids, latent_features, special_token_positions)

    def _forward_training(self, input_ids: torch.Tensor, latent_features: torch.Tensor,
                          special_token_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Training forward pass without caching"""
        batch_size, seq_len = input_ids.shape

        # Get fresh token embeddings
        token_embeddings = self.base_model.tok_embeddings(input_ids)

        # Encode latent features
        latent_embeddings = self.latent_encoder(latent_features)
        
        # Handle different shapes of latent_embeddings
        if len(latent_embeddings.shape) == 2:  # (batch_size, embedding_dim)
            latent_embeddings = latent_embeddings.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        # Expand latent_embeddings if we need multiple spectral features
        if self.num_spectral_features > 1:
            latent_embeddings = latent_embeddings.repeat(1, self.num_spectral_features, 1)

        # Replace special token embeddings at the BEGINNING of each sequence
        # Since features are now at positions 0 to num_spectral_features-1
        if self.num_spectral_features:
            for batch_idx in range(batch_size):
                # Features are at the beginning, so we always replace positions 0 to num_spectral_features-1
                token_embeddings[batch_idx, :self.num_spectral_features] = latent_embeddings[batch_idx]

        # Simple forward through base model without caching
        out, h = self._simple_forward(token_embeddings)
        return {"logits": out, "h": h}

    def _simple_forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple forward pass for training without caching complications"""
        batch_size, seq_len, embed_dim = embeddings.shape
        device = embeddings.device

        h = embeddings

        # Create fresh RoPE frequencies
        head_dim = self.base_model.params.dim // self.base_model.params.n_heads
        freqs = 1.0 / (self.base_model.params.rope_theta ** (
                    torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        # Create causal mask
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=h.dtype)
            mask = torch.triu(mask, diagonal=1)

        # Pass through transformer layers with simplified attention
        for layer in self.base_model.layers:
            # Pre-norm
            h_norm = layer.attention_norm(h)

            # Attention without caching
            attn_out = self._simple_attention(h_norm, freqs_cis, mask, layer.attention)

            # Residual connection
            h = h + attn_out

            # Feed forward
            ff_norm = layer.ffn_norm(h)
            ff_out = layer.feed_forward(ff_norm)
            h = h + ff_out

        # Final norm and output
        h = self.base_model.norm(h)
        output = self.base_model.output(h).float()

        return output, h

    def _simple_attention(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                          mask: Optional[torch.Tensor], attention_layer) -> torch.Tensor:
        """Simplified attention without KV caching"""
        bsz, seqlen, _ = x.shape

        # Get Q, K, V
        xq = attention_layer.wq(x)
        xk = attention_layer.wk(x)
        xv = attention_layer.wv(x)

        # Reshape for attention
        xq = xq.view(bsz, seqlen, attention_layer.n_local_heads, attention_layer.head_dim)
        xk = xk.view(bsz, seqlen, attention_layer.n_local_kv_heads, attention_layer.head_dim)
        xv = xv.view(bsz, seqlen, attention_layer.n_local_kv_heads, attention_layer.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Repeat KV heads if needed
        keys = repeat_kv(xk, attention_layer.n_rep)
        values = repeat_kv(xv, attention_layer.n_rep)

        # Compute attention
        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(attention_layer.head_dim)

        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)

        # Reshape output
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return attention_layer.wo(output)

    def generate_response_from_batch(self,
                                batch_data: dict,
                                batch_idx: int = 0,
                                tokenizer=None,
                                max_new_tokens: int = 100,
                                temperature: float = 0.7,
                                top_p: float = 0.9) -> tuple:
        """
        Generate response from batch data (designed for trainer evaluation)
        
        Args:
            batch_data: Batch dictionary from dataloader
            batch_idx: Which sample in the batch to use
            tokenizer: Tokenizer for decoding
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            
        Returns:
            (generated_text, input_text, target_text, generation_log_probs)
        """
        self.eval()
        
        # Extract data from batch
        input_ids = batch_data['input_ids'][batch_idx:batch_idx+1]  # [1, seq_len]
        input_spectra = batch_data['masked_spectra'][batch_idx:batch_idx+1]  # [1, ...]
        feature_start_idx = batch_data['feature_start_indices'][batch_idx].item()
        answer_start_idx = batch_data['answer_start_indices'][batch_idx].item()
        
        # Get text for reference
        input_text = batch_data['input_texts'][batch_idx] if 'input_texts' in batch_data else "N/A"
        target_text = batch_data['target_texts'][batch_idx] if 'target_texts' in batch_data else "N/A"
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        input_spectra = input_spectra.to(device)
        
        # Create generation input: features + question only (no answer part)
        # Structure: [FEATURES] + [question] + [padding]
        generation_input = input_ids.clone()
        
        # Truncate at answer start to create question-only input
        if answer_start_idx < generation_input.shape[1]:
            # Keep only up to answer start, then pad
            question_part = generation_input[:, :answer_start_idx]
            padding_needed = generation_input.shape[1] - question_part.shape[1]
            if padding_needed > 0:
                padding = torch.full((1, padding_needed), -100, device=device, dtype=torch.long)
                generation_input = torch.cat([question_part, padding], dim=1)
            else:
                generation_input = question_part
        
        original_length = generation_input.shape[1]
        current_tokens = generation_input.clone()
        
        # Track generation probabilities
        generation_log_probs = []
        generated_token_ids = []
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                special_token_positions = torch.tensor([feature_start_idx], device=device)
                
                outputs = self.forward(
                        input_ids=current_tokens,
                        input_spectra=input_spectra,  # Keep providing features
                        special_token_positions=special_token_positions,
                    )
                
                logits = outputs['logits']
                next_token_logits = logits[0, -1, :]  # Last position
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    
                    # Apply top-p sampling - FIXED VERSION
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # FIXED: Remove ellipsis (...) for 1D tensors
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0
                        
                        # Create a mask for the original tensor positions
                        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                        indices_to_remove[sorted_indices] = sorted_indices_to_remove
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    # Greedy sampling
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                next_token_id = next_token.item()
                generated_token_ids.append(next_token_id)
                
                # Store probability for this token
                token_prob = probs[next_token_id].item()
                if token_prob > 1e-10:
                    generation_log_probs.append(np.log(token_prob))
                
                # Check for stopping conditions
                if tokenizer and hasattr(tokenizer, 'eos_id') and next_token_id == tokenizer.eos_id:
                    break
                if tokenizer and hasattr(tokenizer, 'pad_id') and next_token_id == tokenizer.pad_id:
                    break
                    
                # Append new token
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)
        
        # Decode generated text
        if tokenizer and generated_token_ids:
            try:
                generated_text = tokenizer.decode(generated_token_ids).strip()
            except Exception as e:
                generated_text = f"[Decode error: {e}]"
        else:
            generated_text = f"[Token IDs: {generated_token_ids}]"
        
        return generated_text, input_text, target_text, generation_log_probs



def create_multimodal_llama(base_model, latent_dim: int, out_dim: int,  tokenizer, special_token: str = "STAR"):
    """
    Create a multimodal LLaMA model

    Args:
        base_model: The base LLaMA transformer model
        latent_dim: Dimension of latent features
        out_dim: Dimension of output features for numerical regression tasks
        tokenizer: The tokenizer (to get special token ID)
        special_token: Special token string

    Returns:
        MultimodalLlamaModel instance
    """
    # Create multimodal model
    model = MultimodalLlamaModel(base_model, latent_dim, out_dim, special_token)

    # Ensure special token is in tokenizer vocabulary
    # if special_token not in tokenizer.special_tokens:
    #     # Add special token to tokenizer
    #     new_token_id = max(tokenizer.special_tokens.values()) + 1
    #     tokenizer.special_tokens[special_token] = new_token_id
    #     print(f"Added special token '{special_token}' with ID: {new_token_id}")

    #     # Extend model vocabulary if needed
    #     current_vocab_size = base_model.params.vocab_size
    #     if new_token_id >= current_vocab_size:
    #         # Need to extend embeddings and output layer
    #         new_vocab_size = new_token_id + 100  # Add some buffer

    #         # Extend token embeddings
    #         old_embeddings = base_model.tok_embeddings
    #         new_embeddings = nn.Embedding(new_vocab_size, base_model.params.dim)

    #         with torch.no_grad():
    #             new_embeddings.weight[:current_vocab_size] = old_embeddings.weight
    #             # Initialize new tokens randomly
    #             nn.init.normal_(new_embeddings.weight[current_vocab_size:], std=0.02)

    #         base_model.tok_embeddings = new_embeddings

    #         # Extend output layer
    #         old_output = base_model.output
    #         new_output = nn.Linear(base_model.params.dim, new_vocab_size, bias=False)

    #         with torch.no_grad():
    #             new_output.weight[:current_vocab_size] = old_output.weight
    #             # Initialize new tokens randomly
    #             nn.init.normal_(new_output.weight[current_vocab_size:], std=0.02)

    #         base_model.output = new_output
    #         base_model.params.vocab_size = new_vocab_size

    #         print(f"Extended vocabulary from {current_vocab_size} to {new_vocab_size}")

    # Set special token ID
    # model.set_special_token_id(tokenizer.special_tokens[special_token])

    return model


def train_step(model: 'MultimodalLlamaModel',
               batch: dict,
               optimizer: torch.optim.Optimizer,
               device: str = 'cuda',
               ) -> Dict[str, torch.Tensor]:
    """
    Modified training step with stellar parameter focused loss

    Args:
        model: The multimodal model
        batch: Batch from dataloader
        optimizer: Optimizer
        device: Device to run on
        loss_focus: 'parameters_only', 'weighted', or 'standard'
        param_weight: Weight for parameter tokens (when loss_focus='weighted')
        other_weight: Weight for non-parameter tokens (when loss_focus='weighted')

    Returns:
        Loss value
    """
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    # target_ids = batch['target_ids'].to(device)
    # loss_mask = batch['loss_mask'].to(device)
    latent_features = batch['latent_features'].to(device)
    special_token_positions = batch['special_token_positions'].to(device)

    # Check if model has trainable parameters
    # trainable_params = [p for p in model.parameters() if p.requires_grad]
    # if not trainable_params:
    #     print("Warning: No trainable parameters found!")
    #     return {'logits':torch.zeros(0, device=device)}

    # Forward pass
    out = model(input_ids, latent_features, special_token_positions, start_pos=0)

    return out



# Example usage
def example_usage():
    """Example of how to use the multimodal model"""

    # This would be your actual model setup
    # base_model = load_your_llama_model()
    # tokenizer = load_your_tokenizer()
    # latent_dim = 128

    # model = create_multimodal_llama(base_model, latent_dim, tokenizer)

    # Example question about a star
    # question = "Describe the physical parameters of this star <STAR_DATA>"
    # question_tokens = tokenizer.encode(question, bos=True, eos=False)

    # Example latent features (normalized)
    # latent_features = torch.randn(latent_dim)

    # Generate response
    # response = model.generate_response(
    #     question_tokens, latent_features, tokenizer,
    #     max_new_tokens=50, temperature=0.7
    # )
    # print(f"Question: {question}")
    # print(f"Response: {response}")

    pass


if __name__ == "__main__":
    example_usage()