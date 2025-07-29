import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
import torch.nn as nn


def get_stellar_type(temperature, luminosity):
    """
    Returns stellar type based on temperature (K) and luminosity (solar luminosities).

    Args:
        temperature (float): Effective temperature in Kelvin
        luminosity (float): Luminosity in solar luminosities

    Returns:
        str: Stellar classification (O, B, A, F, G, K, or M)
    """

    # Check each stellar class from hottest to coolest
    # Using luminosity ranges from the table
    if temperature >= 33000 and luminosity >= 30000:
        return "O"

    elif 10000 <= temperature < 33000 and 25 <= luminosity <= 30000:
        return "B"

    elif 7300 <= temperature < 10000 and 5 <= luminosity <= 25:
        return "A"

    elif 6000 <= temperature < 7300 and 1.5 <= luminosity <= 5:
        return "F"

    elif 5300 <= temperature < 6000 and 0.6 <= luminosity <= 1.5:
        return "G"

    elif 3900 <= temperature < 5300 and 0.08 <= luminosity <= 0.6:
        return "K"

    elif 2300 <= temperature < 3900 and luminosity <= 0.08:
        return "M"

    else:
        return "Unknown"  # Outside normal main sequence ranges

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


class StarDataset(Dataset):
    """Dataset for star QA with latent features and text descriptions"""

    def __init__(self,
                 latent_features: np.ndarray,
                 metadata_df: pd.DataFrame,
                 llama_tokenizer,
                 max_sequence_length: int = 512,
                 special_token: str = "<STAR_DATA>",
                 noise_std: float = 0.01):
        """
        Args:
            latent_features: Array of shape (n_samples, latent_dim)
            metadata_df: DataFrame with columns like 'Teff', 'logg', 'type', etc.
            llama_tokenizer: LLaMA tokenizer
            max_sequence_length: Maximum sequence length for the model
            special_token: Special token to mark where latent data should be inserted
            noise_std: Standard deviation of Gaussian noise to add to features
        """
        assert len(latent_features) == len(metadata_df), "Features and metadata must have same length"

        self.latent_features = latent_features
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.llama_tokenizer = llama_tokenizer
        self.max_sequence_length = max_sequence_length
        self.special_token = special_token
        self.noise_std = noise_std

        # Add special token to LLaMA tokenizer if not present
        if special_token not in llama_tokenizer.special_tokens:
            self.special_token_id = max(llama_tokenizer.special_tokens.values()) + 1
        else:
            self.special_token_id = llama_tokenizer.special_tokens[special_token]

        # Question templates - asking about the star
        self.question_templates = [
            f"Describe the physical parameters of this star {special_token}",
            f"What are the characteristics of this stellar object {special_token}?",
            f"Analyze the properties of this star {special_token}",
            f"Tell me about this star's temperature and gravity {special_token}",
            f"What type of star is this {special_token}?",
            f"Provide the stellar classification for {special_token}",
            f"Characterize this stellar object {special_token}",
            f"What are the main properties of this star {special_token}?",
            f"Describe the stellar parameters of {special_token}",
            f"Give me the physical characteristics of this star {special_token}",
        ]

        # Answer templates - describing the star
        self.answer_templates = [
            "This star has an effective temperature of {Teff:.0f} K, Luminosity {Lstar:.2f} and surface gravity log(g) = {logg:.2f}. It is classified as a {type} star.",
            "The stellar object shows Teff = {Teff:.0f} K, Luminosity = {Lstar:.2f}, and logg = {logg:.2f}, and belongs to the {type} spectral class.",
            "This {type} star exhibits a temperature of {Teff:.0f} K with log(g) = {logg:.2f} and Luminosity = {Lstar:.2f}.",
            "Physical parameters: effective temperature {Teff:.0f} K, surface gravity log(g) {logg:.2f}, Luminosity {Lstar:.2f}. Spectral type: {type}.",
            "The star is a {type} with Teff = {Teff:.0f} K, Luminosity = {Lstar:.2f} and logg = {logg:.2f}.",
            "Stellar classification: {type}. Temperature: {Teff:.0f} K. Luminosity = {Lstar:.2f}. Surface gravity: log(g) = {logg:.2f}.",
            "This is a {type} star characterized by Teff = {Teff:.0f} K, Luminosity = {Lstar:.2f}, and log(g) = {logg:.2f}.",
            "The object is classified as a {type} star with effective temperature {Teff:.0f} K, Luminosity = {Lstar:.2f} and surface gravity log(g) = {logg:.2f}.",
            "Properties: {type} spectral class, {Teff:.0f} K effective temperature, {Lstar:.2f} Luminosity, log(g) = {logg:.2f}.",
            "This stellar object belongs to the {type} class with Teff = {Teff:.0f} K, Luminosity = {Lstar:.2f}, and logg = {logg:.2f}.",
        ]
    def __len__(self) -> int:
        return len(self.latent_features)

    def _add_noise_to_features(self, features: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to latent features"""
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, features.shape)
            return features + noise
        return features

    def _find_special_token_position(self, tokens: List[int]) -> Optional[int]:
        """Find the position of the special token in the token sequence"""

        # Method 1: Try to find by tokenizing the special token separately
        try:
            special_token_ids = self.llama_tokenizer.encode(self.special_token, bos=False, eos=False)

            # Look for this sequence in the tokens
            for i in range(len(tokens) - len(special_token_ids) + 1):
                if tokens[i:i + len(special_token_ids)] == special_token_ids:
                    return i
        except Exception as e:
            print(f"Method 1 failed: {e}")

        # Method 2: Decode each token and look for the special token string
        for i, token_id in enumerate(tokens):
            try:
                decoded = self.llama_tokenizer.decode([token_id])
                if self.special_token in decoded:
                    return i
            except Exception as e:
                continue

        # Method 3: Decode the entire sequence and try to find approximate position
        try:
            full_text = self.llama_tokenizer.decode(tokens)
            if self.special_token in full_text:
                # Find the character position
                char_pos = full_text.find(self.special_token)

                # Rough approximation: assume each token is about 3-4 characters
                approx_token_pos = min(char_pos // 4, len(tokens) - 1)
                return approx_token_pos
        except Exception as e:
            print(f"Method 3 failed: {e}")

        # If all methods fail, print debug info
        print(f"Could not find special token '{self.special_token}' in sequence")
        print(f"Tokens: {tokens[:10]}...{tokens[-10:]} (showing first/last 10)")
        try:
            decoded_text = self.llama_tokenizer.decode(tokens)
            print(f"Decoded text: '{decoded_text[:100]}...{decoded_text[-100:]}'")
        except:
            print("Could not decode tokens for debugging")

        return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get data for this sample
        latent_vector = self.latent_features[idx]
        metadata = self.metadata_df.iloc[idx]

        # Add noise to features
        noisy_features = self._add_noise_to_features(latent_vector)

        # Choose random templates
        question_template = random.choice(self.question_templates)
        answer_template = random.choice(self.answer_templates)

        # Format question and answer
        question = question_template
        answer = answer_template.format(
            Teff=metadata['Teff'],
            Lstar=np.exp(metadata['Lstar']),
            logg=metadata['logg'],
            type=get_stellar_type(metadata['Teff'], np.exp(metadata['Lstar']))
        )

        # Tokenize question and answer
        question_tokens = self.llama_tokenizer.encode(question, bos=True, eos=False)
        answer_tokens = self.llama_tokenizer.encode(answer, bos=False, eos=True)

        # Find special token position in question
        special_token_pos = self._find_special_token_position(question_tokens)

        # Create full sequence: question + answer
        full_sequence = question_tokens + answer_tokens

        # Calculate where special token is in the full sequence
        if special_token_pos is not None:
            full_special_token_pos = special_token_pos
        else:
            # If not found, set to -1 (will be handled in model)
            full_special_token_pos = -1

        # Truncate if too long
        if len(full_sequence) > self.max_sequence_length:
            full_sequence = full_sequence[:self.max_sequence_length]

        # For training: input is question + answer[:-1], target is question[1:] + answer
        # But we only want to compute loss on the answer part
        question_len = len(question_tokens)

        # Create input and target sequences
        input_sequence = full_sequence[:-1]  # All except last token
        target_sequence = full_sequence[1:]  # All except first token

        # Create loss mask - only compute loss on answer tokens
        loss_mask = torch.zeros(len(target_sequence), dtype=torch.bool)
        if question_len < len(target_sequence):
            loss_mask[question_len - 1:] = True  # Only answer part

        return {
            'input_ids': torch.tensor(input_sequence, dtype=torch.long),
            'target_ids': torch.tensor(target_sequence, dtype=torch.long),
            'loss_mask': loss_mask,
            'latent_features': torch.tensor(noisy_features, dtype=torch.float32),
            'special_token_pos': torch.tensor(full_special_token_pos, dtype=torch.long),
            'question_text': question,
            'answer_text': answer,
            'metadata': {key: metadata[key] for key in metadata.index}
        }

    def debug_special_token(self, sample_idx: int = 0):
        """Debug special token detection for a specific sample"""
        print(f"\n=== Debug Special Token (Sample {sample_idx}) ===")

        # Get sample data
        latent_vector = self.latent_features[sample_idx]
        metadata = self.metadata_df.iloc[sample_idx]

        # Create question with special token
        question_template = f"Describe the physical parameters of this star {self.special_token}"
        print(f"Question template: '{question_template}'")

        # Tokenize
        question_tokens = self.llama_tokenizer.encode(question_template, bos=True, eos=False)
        print(f"Question tokens: {question_tokens}")
        print(f"Number of tokens: {len(question_tokens)}")

        # Try to decode back
        try:
            decoded = self.llama_tokenizer.decode(question_tokens)
            print(f"Decoded back: '{decoded}'")
        except Exception as e:
            print(f"Decode failed: {e}")

        # Test special token tokenization
        print(f"\nSpecial token: '{self.special_token}'")
        try:
            special_only = self.llama_tokenizer.encode(self.special_token, bos=False, eos=False)
            print(f"Special token alone: {special_only}")

            special_decoded = self.llama_tokenizer.decode(special_only)
            print(f"Special token decoded: '{special_decoded}'")
        except Exception as e:
            print(f"Special token encoding failed: {e}")

        # Test finding position
        found_pos = self._find_special_token_position(question_tokens)
        print(f"Found position: {found_pos}")

        if found_pos is not None:
            print(f"Token at found position: {question_tokens[found_pos]}")
            try:
                token_decoded = self.llama_tokenizer.decode([question_tokens[found_pos]])
                print(f"Token at position decoded: '{token_decoded}'")
            except:
                print("Could not decode token at position")

        # Test if special token is in tokenizer vocabulary
        if hasattr(self.llama_tokenizer, 'special_tokens'):
            if self.special_token in self.llama_tokenizer.special_tokens:
                token_id = self.llama_tokenizer.special_tokens[self.special_token]
                print(f"Special token is in vocabulary with ID: {token_id}")
            else:
                print(f"Special token '{self.special_token}' NOT in tokenizer vocabulary")
                print(f"Available special tokens: {list(self.llama_tokenizer.special_tokens.keys())}")

        print("=== End Debug ===\n")


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle variable length sequences"""

    # Get maximum length in this batch
    max_len = max(len(item['input_ids']) for item in batch)

    # Pad sequences
    input_ids = []
    target_ids = []
    loss_masks = []
    special_token_positions = []

    for item in batch:
        input_seq = item['input_ids']
        target_seq = item['target_ids']
        loss_mask = item['loss_mask']

        # Pad input sequence
        pad_length = max_len - len(input_seq)
        padded_input = torch.cat([
            input_seq,
            torch.zeros(pad_length, dtype=torch.long)
        ])

        # Pad target sequence
        padded_target = torch.cat([
            target_seq,
            torch.full((pad_length,), -100, dtype=torch.long)  # -100 is ignore index for loss
        ])

        # Pad loss mask
        padded_loss_mask = torch.cat([
            loss_mask,
            torch.zeros(pad_length, dtype=torch.bool)
        ])

        input_ids.append(padded_input)
        target_ids.append(padded_target)
        loss_masks.append(padded_loss_mask)
        special_token_positions.append(item['special_token_pos'])

    return {
        'input_ids': torch.stack(input_ids),
        'target_ids': torch.stack(target_ids),
        'loss_mask': torch.stack(loss_masks),
        'latent_features': torch.stack([item['latent_features'] for item in batch]),
        'special_token_positions': torch.stack(special_token_positions),
        'question_texts': [item['question_text'] for item in batch],
        'answer_texts': [item['answer_text'] for item in batch]
    }


# Example usage and testing
def create_sample_data(n_samples: int = 1000, latent_dim: int = 128):
    """Create sample data for testing"""

    # Generate fake latent features
    latent_features = np.random.randn(n_samples, latent_dim)

    # Generate fake metadata
    star_types = ['G-dwarf', 'K-giant', 'M-dwarf', 'F-main', 'A-star', 'B-star']

    metadata = {
        'Teff': np.random.uniform(3000, 10000, n_samples),  # Temperature in K
        'logg': np.random.uniform(1.0, 5.0, n_samples),  # Surface gravity
        'type': np.random.choice(star_types, n_samples)
    }

    metadata_df = pd.DataFrame(metadata)

    return latent_features, metadata_df


def setup_dataloaders(latent_features: np.ndarray,
                      metadata_df: pd.DataFrame,
                      llama_tokenizer,
                      batch_size: int = 8,
                      train_split: float = 0.8,
                      noise_std: float = 0.01):
    """Setup train and validation dataloaders"""

    # Split data
    n_train = int(len(latent_features) * train_split)

    train_features = latent_features[:n_train]
    train_metadata = metadata_df.iloc[:n_train]

    val_features = latent_features[n_train:]
    val_metadata = metadata_df.iloc[n_train:]

    # Create datasets
    train_dataset = StarDataset(
        train_features, train_metadata, llama_tokenizer, noise_std=noise_std
    )

    val_dataset = StarDataset(
        val_features, val_metadata, llama_tokenizer, noise_std=0.0  # No noise for validation
    )

    # Create dataloaders with num_workers=0 to avoid pickling issues on Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Changed from 2 to 0
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Changed from 2 to 0
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    print("Creating sample data...")
    latent_features, metadata_df = create_sample_data(n_samples=100, latent_dim=128)

    print(f"Created {len(latent_features)} samples")
    print(f"Latent features shape: {latent_features.shape}")
    print(f"Metadata columns: {list(metadata_df.columns)}")
    print(f"Sample metadata:\n{metadata_df.head()}")

    # Example of a single sample
    from llama3.llama.tokenizer import Tokenizer  # This would need your actual tokenizer
    # tokenizer = Tokenizer(model_path="path/to/tokenizer.model")
    # dataset = StarDataset(latent_features, metadata_df, tokenizer)
    # sample = dataset[0]
    # print(f"\nSample structure:")
    # for key, value in sample.items():
    #     if isinstance(value, torch.Tensor):
    #         print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
    #     else:
    #         print(f"  {key}: {value}")