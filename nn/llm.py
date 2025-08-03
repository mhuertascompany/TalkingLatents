import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

TOKENIZER_PATH = r"C:\Users\Ilay\.llama\checkpoints\Llama3.2-1B\tokenizer.model"
MODEL_PATH = r"C:\Users\Ilay\.llama\checkpoints\Llama3.2-1B"

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

    def __init__(self, base_model, latent_dim: int, out_dim: int, special_token: str = " STAR"):
        super().__init__()
        self.base_model = base_model
        self.embedding_dim = base_model.params.dim
        self.special_token = special_token
        self.latent_regressor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 4),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim // 4),
            nn.Linear(self.embedding_dim // 4, out_dim)
        )

        # Latent feature encoder - projects features to embedding space
        self.latent_encoder = LatentFeatureEncoder(
            latent_dim=latent_dim,
            embedding_dim=self.embedding_dim,
            hidden_dim=512
        )

        # Get special token ID
        # This should be set when you initialize with the tokenizer
        self.special_token_id = None

    def set_special_token_id(self, token_id: int):
        """Set the special token ID after tokenizer is available"""
        self.special_token_id = token_id

    def forward(self,
                input_ids: torch.Tensor,
                latent_features: torch.Tensor,
                special_token_positions: torch.Tensor,
                start_pos: int = 0) -> Dict[str, torch.Tensor]:
        """
        Forward pass with latent feature integration at embedding level
        """
        batch_size, seq_len = input_ids.shape

        # For training, always use start_pos=0 and bypass caching
        if self.training:
            return self._forward_training(input_ids, latent_features, special_token_positions)
        else:
            return self._forward_generation(input_ids, latent_features, special_token_positions, start_pos)

    def _forward_training(self, input_ids: torch.Tensor, latent_features: torch.Tensor,
                          special_token_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Training forward pass without caching"""
        batch_size, seq_len = input_ids.shape

        # Get fresh token embeddings
        token_embeddings = self.base_model.tok_embeddings(input_ids)

        # Encode latent features
        latent_embeddings = self.latent_encoder(latent_features)

        # Replace special token embeddings
        for batch_idx in range(batch_size):
            special_pos = special_token_positions[batch_idx].item()
            if 0 <= special_pos < seq_len:
                token_embeddings[batch_idx, special_pos] = latent_embeddings[batch_idx]

        # Simple forward through base model without caching
        out, h = self._simple_forward(token_embeddings)
        reg_out = self.latent_regressor(latent_features)
        return {"logits": out, "h": h, "reg_out": reg_out}

    def _forward_generation(self, input_ids: torch.Tensor, latent_features: torch.Tensor,
                            special_token_positions: torch.Tensor, start_pos: int) -> Dict[str, torch.Tensor]:
        """Generation forward pass with caching (for inference)"""
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        token_embeddings = self.base_model.tok_embeddings(input_ids)

        # Encode latent features
        latent_embeddings = self.latent_encoder(latent_features)

        # Replace special token embeddings
        modified_embeddings = token_embeddings.clone()
        for batch_idx in range(batch_size):
            special_pos = special_token_positions[batch_idx].item()
            if 0 <= special_pos < seq_len:
                modified_embeddings[batch_idx, special_pos] = latent_embeddings[batch_idx]

        out, h = self._forward_with_embeddings(modified_embeddings, start_pos)
        reg_out = self.latent_regressor(latent_features)
        return {"logits": out, "h": h, "reg_out": reg_out}

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

    def _forward_with_embeddings(self, embeddings: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the transformer with custom embeddings
        """
        _bsz, seqlen, _ = embeddings.shape
        h = embeddings

        # Get RoPE frequencies - create fresh copy to avoid caching issues
        device = h.device
        freqs_cis = precompute_freqs_cis(
            self.base_model.params.dim // self.base_model.params.n_heads,
            seqlen + start_pos,
            self.base_model.params.rope_theta,
        ).to(device)

        freqs_cis = freqs_cis[start_pos: start_pos + seqlen]

        # Create attention mask if needed
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=h.dtype)
            mask = torch.triu(mask, diagonal=1)

            if start_pos > 0:
                mask = torch.hstack(
                    [torch.zeros((seqlen, start_pos), device=device, dtype=h.dtype), mask]
                )

        # Pass through transformer layers
        for layer in self.base_model.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        # Final layer norm and output projection
        h = self.base_model.norm(h)
        output = self.base_model.output(h).float()

        return output, h

    def generate_response(self,
                          question_tokens: List[int],
                          latent_features: torch.Tensor,
                          tokenizer,
                          special_token_pos: int,
                          max_new_tokens: int = 100,
                          temperature: float = 0.7,
                          top_p: float = 0.9,
                          device: str = 'cuda') -> str:
        """
        Generate a response to a question about the star

        Args:
            question_tokens: List of token IDs for the question
            latent_features: (latent_dim,) features for this star
            tokenizer: The tokenizer
            special_token_pos: Special token position
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            device: Device to use

        Returns:
            Generated response text
        """
        # self.eval()

        # Find special token position using improved method
        # special_token_pos = self._find_special_token_position(question_tokens, tokenizer)

        print(f"Debug: Special token position found at index {special_token_pos}")
        if special_token_pos < len(question_tokens):
            token_at_pos = tokenizer.decode([question_tokens[special_token_pos]])
            print(f"Debug: Token at position {special_token_pos}: '{token_at_pos}'")

        # Convert to tensors
        current_tokens = torch.tensor([question_tokens], dtype=torch.long, device=device)
        latent_batch = latent_features.unsqueeze(0).to(device)  # Add batch dimension
        special_pos_batch = torch.tensor([special_token_pos], dtype=torch.long, device=device)

        # Generate tokens one by one
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self.forward(
                    current_tokens,
                    latent_batch,
                    special_pos_batch,
                    start_pos=0
                )

                # Get next token probabilities
                next_token_logits = logits[0, -1, :]  # Last token of first batch

                if temperature > 0:
                    # Apply temperature
                    next_token_logits = next_token_logits / temperature

                    # Apply top-p sampling
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = self._sample_top_p(probs, top_p)
                else:
                    # Greedy sampling
                    next_token = torch.argmax(next_token_logits, dim=-1)

                # Check for stop tokens
                if next_token.item() in tokenizer.stop_tokens:
                    break

                # Append new token
                next_token = next_token.unsqueeze(0)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)

        # Decode the full sequence
        generated_tokens = current_tokens[0].tolist()
        full_text = tokenizer.decode(generated_tokens)

        # Extract just the answer part (everything after the question)
        question_text = tokenizer.decode(question_tokens)
        if question_text in full_text:
            answer_text = full_text[len(question_text):].strip()
        else:
            answer_text = full_text

        return answer_text

    # def _find_special_token_position(self, question_tokens: List[int], tokenizer) -> int:
    #     """
    #     Find the position of the special token in the question tokens
    #     Handles multi-token special tokens that get split by the tokenizer
    #     """
    #     # Method 1: Try to find the exact special token ID if it exists as single token
    #     try:
    #         special_token_id = tokenizer.encode(self.special_token, bos=False, eos=False)[0]
    #         if special_token_id in question_tokens:
    #             return question_tokens.index(special_token_id)
    #     except:
    #         pass
    #
    #     # Method 2: Reconstruct text and find special token position
    #     try:
    #         # Decode the full question
    #         question_text = tokenizer.decode(question_tokens)
    #
    #         # Find where the special token appears in the text
    #         special_token_start = question_text.find(self.special_token)
    #         if special_token_start != -1:
    #             # Find which token position corresponds to this character position
    #             char_count = 0
    #             for i, token_id in enumerate(question_tokens):
    #                 token_text = tokenizer.decode([token_id])
    #                 if char_count <= special_token_start < char_count + len(token_text):
    #                     return i
    #                 char_count += len(token_text)
    #     except:
    #         pass
    #
    #     # Method 3: Look for token sequences that spell out the special token
    #     special_token_parts = tokenizer.encode(self.config['special_token'], bos=False, eos=False)
    #     if len(special_token_parts) > 1:
    #         # Look for the sequence of tokens
    #         for i in range(len(question_tokens) - len(special_token_parts) + 1):
    #             if question_tokens[i:i + len(special_token_parts)] == special_token_parts:
    #                 return i  # Return position of first token in sequence
    #
    #     # Method 4: Fallback - look for partial matches
    #     for i, token_id in enumerate(question_tokens):
    #         try:
    #             decoded = tokenizer.decode([token_id])
    #             if any(part in decoded for part in ['<STAR', 'STAR_', '_DATA', 'DATA>']):
    #                 return i
    #         except:
    #             continue
    #
    #     print(f"Warning: Special token '{self.special_token}' not found in question")
    #     print(f"Question tokens: {question_tokens}")
    #     print(f"Decoded tokens: {[tokenizer.decode([t]) for t in question_tokens]}")
    #     return len(question_tokens) - 1  # Use last position as fallback

    def _sample_top_p(self, probs: torch.Tensor, p: float) -> torch.Tensor:
        """Top-p (nucleus) sampling"""
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


def create_multimodal_llama(base_model, latent_dim: int, out_dim: int,  tokenizer, special_token: str = " STAR"):
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
    if special_token not in tokenizer.special_tokens:
        # Add special token to tokenizer
        new_token_id = max(tokenizer.special_tokens.values()) + 1
        tokenizer.special_tokens[special_token] = new_token_id
        print(f"Added special token '{special_token}' with ID: {new_token_id}")

        # Extend model vocabulary if needed
        current_vocab_size = base_model.params.vocab_size
        if new_token_id >= current_vocab_size:
            # Need to extend embeddings and output layer
            new_vocab_size = new_token_id + 100  # Add some buffer

            # Extend token embeddings
            old_embeddings = base_model.tok_embeddings
            new_embeddings = nn.Embedding(new_vocab_size, base_model.params.dim)

            with torch.no_grad():
                new_embeddings.weight[:current_vocab_size] = old_embeddings.weight
                # Initialize new tokens randomly
                nn.init.normal_(new_embeddings.weight[current_vocab_size:], std=0.02)

            base_model.tok_embeddings = new_embeddings

            # Extend output layer
            old_output = base_model.output
            new_output = nn.Linear(base_model.params.dim, new_vocab_size, bias=False)

            with torch.no_grad():
                new_output.weight[:current_vocab_size] = old_output.weight
                # Initialize new tokens randomly
                nn.init.normal_(new_output.weight[current_vocab_size:], std=0.02)

            base_model.output = new_output
            base_model.params.vocab_size = new_vocab_size

            print(f"Extended vocabulary from {current_vocab_size} to {new_vocab_size}")

    # Set special token ID
    model.set_special_token_id(tokenizer.special_tokens[special_token])

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
    target_ids = batch['target_ids'].to(device)
    loss_mask = batch['loss_mask'].to(device)
    latent_features = batch['latent_features'].to(device)
    special_token_positions = batch['special_token_positions'].to(device)

    # Check if model has trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("Warning: No trainable parameters found!")
        return {'logits':torch.zeros(0, device=device)}

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