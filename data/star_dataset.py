import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import random
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import torch.nn as nn
import re
import time


def identify_stellar_parameter_tokens(tokenizer, answer_text: str, answer_tokens: List[int]) -> List[bool]:
    """
    Identify which tokens in the answer correspond to stellar parameter values

    Args:
        tokenizer: The tokenizer used
        answer_text: The full answer text
        answer_tokens: List of token IDs for the answer

    Returns:
        List of booleans indicating which tokens are stellar parameter values
    """
    # Patterns to match stellar parameter values
    patterns = [
        r'(?:Teff[=\s]*|temperature[=\s]*|T[=\s]*)?(\d+(?:\.\d+)?)\s*K',  # Temperature
        r'(?:logg[=\s]*|log\(g\)[=\s]*|gravity[=\s]*)?(\d+(?:\.\d+)?)',  # Surface gravity
        r'(?:Lstar[=\s]*|L[=\s]*|luminosity[=\s]*)?(\d+(?:\.\d+)?)',  # Luminosity
        r'(\d+(?:\.\d+)?)\s*(?:solar|L_sun|L☉)',  # Solar luminosity units
    ]

    # Find all parameter value positions in text
    param_positions = []
    for pattern in patterns:
        for match in re.finditer(pattern, answer_text, re.IGNORECASE):
            start, end = match.span(1) if match.lastindex else match.span()
            param_positions.append((start, end))

    # Map character positions to token positions
    token_mask = [False] * len(answer_tokens)
    char_pos = 0

    for i, token_id in enumerate(answer_tokens):
        try:
            token_text = tokenizer.decode([token_id])
            token_start = char_pos
            token_end = char_pos + len(token_text)

            # Check if this token overlaps with any parameter value
            for param_start, param_end in param_positions:
                if (token_start < param_end and token_end > param_start):
                    token_mask[i] = True
                    break

            char_pos = token_end
        except:
            # Skip problematic tokens
            continue

    return token_mask

def get_stellar_type(temperature, luminosity):
    """Returns stellar type based on temperature (K) and luminosity (solar luminosities)."""
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
        return "Unknown"


class LatentFeatureAnalyzer:
    """Utility class to analyze latent features and generate interpretable descriptions"""

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.temperature_correlations = {}  # Could be learned from data
        self.gravity_correlations = {}

    def analyze_features(self, latent_features: np.ndarray) -> Dict[str, any]:
        """Analyze latent features and return interpretable properties"""

        # Basic statistics
        mean_activation = float(np.mean(latent_features))
        std_activation = float(np.std(latent_features))
        max_activation = float(np.max(latent_features))
        min_activation = float(np.min(latent_features))

        # Find most active dimensions
        abs_features = np.abs(latent_features)
        top_5_dims = np.argsort(abs_features)[-5:].tolist()
        top_5_values = abs_features[top_5_dims].tolist()

        # Find least active dimensions
        bottom_5_dims = np.argsort(abs_features)[:5].tolist()
        bottom_5_values = abs_features[bottom_5_dims].tolist()

        # Sparsity analysis
        near_zero_count = int(np.sum(abs_features < 0.1))
        sparsity_ratio = near_zero_count / len(latent_features)

        # Activity patterns
        positive_dims = int(np.sum(latent_features > 0.5))
        negative_dims = int(np.sum(latent_features < -0.5))

        # Clustering/grouping (simple version)
        high_activity_regions = []
        for i in range(0, len(latent_features), 10):  # Group in chunks of 10
            chunk = latent_features[i:i + 10]
            if np.mean(np.abs(chunk)) > mean_activation + std_activation:
                high_activity_regions.append(f"{i}-{min(i + 9, len(latent_features) - 1)}")

        return {
            'mean_activation': mean_activation,
            'std_activation': std_activation,
            'max_activation': max_activation,
            'min_activation': min_activation,
            'top_5_dims': top_5_dims,
            'top_5_values': top_5_values,
            'bottom_5_dims': bottom_5_dims,
            'bottom_5_values': bottom_5_values,
            'sparsity_ratio': sparsity_ratio,
            'positive_dims': positive_dims,
            'negative_dims': negative_dims,
            'high_activity_regions': high_activity_regions,
            'activation_range': max_activation - min_activation
        }

    def generate_latent_description(self, latent_analysis: Dict, stellar_params: Dict) -> str:
        """Generate natural language description of latent features in context of stellar parameters"""

        descriptions = []

        # Overall activation pattern
        if latent_analysis['mean_activation'] > 0.5:
            descriptions.append("strong overall feature activation")
        elif latent_analysis['mean_activation'] < -0.5:
            descriptions.append("predominantly negative feature responses")
        else:
            descriptions.append("balanced feature activation")

        # Sparsity information
        if latent_analysis['sparsity_ratio'] > 0.7:
            descriptions.append("sparse representation with few active features")
        elif latent_analysis['sparsity_ratio'] < 0.3:
            descriptions.append("dense representation with many active features")

        # Top dimensions
        top_dims_str = ", ".join(map(str, latent_analysis['top_5_dims'][:3]))
        descriptions.append(f"peak responses in dimensions {top_dims_str}")

        # Regional activity
        if latent_analysis['high_activity_regions']:
            regions_str = ", ".join(latent_analysis['high_activity_regions'][:2])
            descriptions.append(f"concentrated activity in regions {regions_str}")

        # Connect to stellar parameters (this is where interpretability magic happens)
        temp = stellar_params.get('Teff', 0)
        if temp > 6000:
            descriptions.append("consistent with hot star signatures")
        elif temp < 4000:
            descriptions.append("matching cool star patterns")

        return "; ".join(descriptions)


class StarDataset(Dataset):
    """Enhanced dataset with latent feature descriptions"""

    def __init__(self,
                 latent_features: np.ndarray,
                 metadata_df: pd.DataFrame,
                 llama_tokenizer,
                 max_sequence_length: int = 512,
                 special_token: str = "STAR",
                 noise_std: float = 0.01,
                 include_latent_analysis: bool = False):

        assert len(latent_features) == len(metadata_df), "Features and metadata must have same length"

        self.latent_features = latent_features
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.llama_tokenizer = llama_tokenizer
        self.max_sequence_length = max_sequence_length
        self.special_token = special_token
        self.noise_std = noise_std
        self.include_latent_analysis = include_latent_analysis

        # Initialize latent analyzer
        self.latent_analyzer = LatentFeatureAnalyzer(latent_features.shape[1])

        # Enhanced question templates
        self.question_templates = [
            f"Describe the physical parameters of this star {special_token}",
            f"What are the characteristics of this stellar object {special_token}?",
            f"Analyze the properties of this star {special_token}",
            f"What type of star is this {special_token}?",
            f"Explain the stellar parameters and their encoding for {special_token}",
            f"Describe both the physical properties and latent representation of {special_token}",
            f"What do the encoded features reveal about this star {special_token}?",
            f"Interpret the stellar classification and feature patterns of {special_token}",
        ]

        # Basic answer templates (original)
        self.basic_answer_templates = [
            "This star has an effective temperature of {Teff:.0f} K, Luminosity {Lstar:.2f} and surface gravity log(g) = {logg:.2f}. It is classified as a {stellar_type} star.",
            "Physical parameters: effective temperature {Teff:.0f} K, surface gravity log(g) {logg:.2f}, Luminosity {Lstar:.2f}. Spectral type: {stellar_type}.",
            "Properties: {stellar_type} spectral class, {Teff:.0f} K effective temperature, {Lstar:.2f} Luminosity, log(g) = {logg:.2f}.",
        ]

        # Enhanced templates with latent analysis
        self.enhanced_answer_templates = [
            "This star has Teff={Teff:.0f}K, log(g)={logg:.2f}, Luminosity={Lstar:.2f} ({stellar_type} class). The encoded representation shows {latent_description}.",
            "Stellar classification: {stellar_type}. Temperature: {Teff:.0f} K. Luminosity = {Lstar:.2f}. Surface gravity: log(g) = {logg:.2f}. The latent encoding exhibits {latent_description}.",
            "Physical parameters: Teff={Teff:.0f}K, log(g)={logg:.2f}, L={Lstar:.2f}. Classification: {stellar_type}. Feature analysis reveals {latent_description}.",
            "This {stellar_type} star (T={Teff:.0f}K, log(g)={logg:.2f}, L={Lstar:.2f}) shows {latent_description} in its encoded representation.",
        ]

        self.param_stats = {
            'Teff': {'mean': self.metadata_df['Teff'].mean(), 'std': self.metadata_df['Teff'].std()},
            'logg': {'mean': self.metadata_df['logg'].mean(), 'std': self.metadata_df['logg'].std()},
            'Lstar': {'mean': np.exp(self.metadata_df['Lstar']).mean(), 'std': np.exp(self.metadata_df['Lstar']).std()}
        }
        print(f"Parameter normalization stats: {self.param_stats}")

    def normalize_params(self, teff, logg, lstar):
        """Normalize stellar parameters to zero mean, unit variance"""
        norm_teff = (teff - self.param_stats['Teff']['mean']) / self.param_stats['Teff']['std']
        norm_logg = (logg - self.param_stats['logg']['mean']) / self.param_stats['logg']['std']
        norm_lstar = (lstar - self.param_stats['Lstar']['mean']) / self.param_stats['Lstar']['std']
        return norm_teff, norm_logg, norm_lstar

    def denormalize_params(self, norm_teff, norm_logg, norm_lstar):
        """Denormalize parameters back to original scale"""
        teff = norm_teff * self.param_stats['Teff']['std'] + self.param_stats['Teff']['mean']
        logg = norm_logg * self.param_stats['logg']['std'] + self.param_stats['logg']['mean']
        lstar = norm_lstar * self.param_stats['Lstar']['std'] + self.param_stats['Lstar']['mean']
        return teff, logg, lstar

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

        # Method 1: Look for the exact special token ID
        if hasattr(self.llama_tokenizer, 'get_custom_token_id'):
            try:
                special_token_id = self.llama_tokenizer.get_custom_token_id(self.special_token)
                if special_token_id in tokens:
                    return tokens.index(special_token_id)
            except:
                pass

        # Method 2: Handle space variants - tokenize both with and without leading space
        special_token_variants = [
            self.special_token,  # "STAR"
            f" {self.special_token}",  # " STAR"
            f"_{self.special_token}",  # "_STAR" (sometimes used)
            f"<{self.special_token}>",  # "<STAR>" (if using brackets)
        ]

        for variant in special_token_variants:
            try:
                variant_tokens = self.llama_tokenizer.encode(variant, bos=False, eos=False)

                # Look for this sequence in the tokens
                for i in range(len(tokens) - len(variant_tokens) + 1):
                    if tokens[i:i + len(variant_tokens)] == variant_tokens:
                        # print(f"Found special token variant '{variant}' at position {i}")
                        return i

                # Also check for single token match
                if len(variant_tokens) == 1 and variant_tokens[0] in tokens:
                    pos = tokens.index(variant_tokens[0])
                    # print(f"Found single token variant '{variant}' (ID: {variant_tokens[0]}) at position {pos}")
                    return pos

            except Exception as e:
                continue

        # Method 3: Decode and search by string matching
        try:
            full_text = self.llama_tokenizer.decode(tokens)

            # Look for any variant in the decoded text
            for variant in [self.special_token, f" {self.special_token}"]:
                if variant in full_text:
                    char_pos = full_text.find(variant)

                    # Convert character position to approximate token position
                    # This is a rough estimation
                    approx_token_pos = 0
                    char_count = 0

                    for i, token_id in enumerate(tokens):
                        token_text = self.llama_tokenizer.decode([token_id])
                        if char_count <= char_pos < char_count + len(token_text):
                            # print(f"Found special token by string matching at approximate position {i}")
                            return i
                        char_count += len(token_text)

        except Exception as e:
            print(f"String matching failed: {e}")

        # Method 4: Brute force - decode each token and check for substring
        for i, token_id in enumerate(tokens):
            try:
                decoded = self.llama_tokenizer.decode([token_id])
                if self.special_token in decoded:
                    # print(f"Found special token '{self.special_token}' in token '{decoded}' at position {i}")
                    return i
            except:
                continue

        print(f"Could not find special token '{self.special_token}' in sequence")
        print(f"Tokens: {tokens[:10]}...{tokens[-10:]} (showing first/last 10)")

        # Debug: Show what each token decodes to
        print("Token decode debug:")
        for i, token_id in enumerate(tokens[:15]):  # Show first 15 tokens
            try:
                decoded = self.llama_tokenizer.decode([token_id])
                print(f"  {i}: {token_id} -> '{decoded}'")
            except:
                print(f"  {i}: {token_id} -> <decode failed>")

        return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        # Get data for this sample
        latent_vector = self.latent_features[idx]
        metadata = self.metadata_df.iloc[idx]

        # Add noise to features
        noisy_features = self._add_noise_to_features(latent_vector)

        # Analyze latent features
        stellar_params = {
            'Teff': metadata['Teff'],
            'logg': metadata['logg'],
            'Lstar': np.exp(metadata['Lstar'])
        }

        stellar_type = get_stellar_type(stellar_params['Teff'], stellar_params['Lstar'])

        # Choose question template
        question_template = random.choice(self.question_templates)
        question = question_template

        # Generate answer
        if self.include_latent_analysis and random.random() > 0.3:
            latent_analysis = self.latent_analyzer.analyze_features(latent_vector)
            latent_description = self.latent_analyzer.generate_latent_description(
                latent_analysis, stellar_params
            )
            answer_template = random.choice(self.enhanced_answer_templates)
            answer = answer_template.format(
                Teff=stellar_params['Teff'],
                Lstar=stellar_params['Lstar'],
                logg=stellar_params['logg'],
                stellar_type=stellar_type,
                latent_description=latent_description
            )
        else:
            answer_template = random.choice(self.basic_answer_templates)
            answer = answer_template.format(
                Teff=stellar_params['Teff'],
                Lstar=stellar_params['Lstar'],
                logg=stellar_params['logg'],
                stellar_type=stellar_type
            )

        norm_teff, norm_logg, norm_lstar = self.normalize_params(
            stellar_params['Teff'],
            stellar_params['logg'],
            stellar_params['Lstar']
        )
        numerical_targets = torch.tensor([norm_teff, norm_logg, norm_lstar])

        # Tokenize question and answer
        question_tokens = self.llama_tokenizer.encode(question, bos=True, eos=False)
        answer_tokens = self.llama_tokenizer.encode(answer, bos=False, eos=True)

        # Find special token position
        special_token_pos = self._find_special_token_position(question_tokens)

        # Create full sequence
        full_sequence = question_tokens + answer_tokens
        if special_token_pos is not None:
            full_special_token_pos = special_token_pos
        else:
            full_special_token_pos = -1

        # Truncate if too long
        if len(full_sequence) > self.max_sequence_length:
            full_sequence = full_sequence[:self.max_sequence_length]
            # Adjust answer_tokens if truncated
            remaining_answer_len = len(full_sequence) - len(question_tokens)
            if remaining_answer_len > 0:
                answer_tokens = answer_tokens[:remaining_answer_len]
            else:
                answer_tokens = []

        question_len = len(question_tokens)

        # Create input and target sequences
        input_sequence = full_sequence[:-1]
        target_sequence = full_sequence[1:]

        # Create standard loss mask - only compute loss on answer tokens
        loss_mask = torch.zeros(len(target_sequence), dtype=torch.bool)
        if question_len < len(target_sequence):
            loss_mask[question_len - 1:] = True

        # param_loss_mask = torch.zeros(len(target_sequence), dtype=torch.bool)
        #
        # if len(answer_tokens) > 0 and answer.strip():  # Make sure we have valid answer
        #     # Identify parameter tokens in the answer
        #     param_token_indicators = identify_stellar_parameter_tokens(
        #         self.llama_tokenizer, answer, answer_tokens
        #     )
        #
        #     # Map to target sequence positions
        #     answer_start_in_target = question_len - 1
        #     for i, is_param_token in enumerate(param_token_indicators):
        #         target_pos = answer_start_in_target + i
        #         if target_pos < len(param_loss_mask) and is_param_token:
        #             param_loss_mask[target_pos] = True

        return {
            'input_ids': torch.tensor(input_sequence, dtype=torch.long),
            'target_ids': torch.tensor(target_sequence, dtype=torch.long),
            'loss_mask': loss_mask,  # Original mask (all answer tokens)
            'numerical_targets': numerical_targets,
            'latent_features': torch.tensor(noisy_features, dtype=torch.float32),
            'special_token_positions': torch.tensor(full_special_token_pos, dtype=torch.long),
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

    def debug_enhanced_sample(self, sample_idx: int = 0):
        """Debug enhanced sample generation"""
        print(f"\n=== Enhanced Sample Debug (Sample {sample_idx}) ===")

        latent_vector = self.latent_features[sample_idx]
        metadata = self.metadata_df.iloc[sample_idx]

        # Analyze features
        latent_analysis = self.latent_analyzer.analyze_features(latent_vector)
        stellar_params = {
            'Teff': metadata['Teff'],
            'logg': metadata['logg'],
            'Lstar': np.exp(metadata['Lstar'])
        }

        print(f"Stellar parameters: {stellar_params}")
        print(f"Latent analysis: {latent_analysis}")

        latent_description = self.latent_analyzer.generate_latent_description(
            latent_analysis, stellar_params
        )
        print(f"Generated description: {latent_description}")

        # Generate sample
        sample = self[sample_idx]
        print(f"Question: {sample['question_text']}")
        print(f"Answer: {sample['answer_text']}")

        print("=== End Enhanced Debug ===\n")


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Modified collate function to handle parameter loss masks"""
    max_len = max(len(item['input_ids']) for item in batch)

    input_ids = []
    target_ids = []
    loss_masks = []
    targets = []  # NEW
    special_token_positions = []

    for item in batch:
        input_seq = item['input_ids']
        target_seq = item['target_ids']
        loss_mask = item['loss_mask']
        numerical_targets = item['numerical_targets']

        pad_length = max_len - len(input_seq)

        padded_input = torch.cat([input_seq, torch.zeros(pad_length, dtype=torch.long)])
        padded_target = torch.cat([target_seq, torch.full((pad_length,), -100, dtype=torch.long)])
        padded_loss_mask = torch.cat([loss_mask, torch.zeros(pad_length, dtype=torch.bool)])

        input_ids.append(padded_input)
        target_ids.append(padded_target)
        loss_masks.append(padded_loss_mask)
        targets.append(numerical_targets)
        special_token_positions.append(item['special_token_positions'])

    return {
        'input_ids': torch.stack(input_ids),
        'target_ids': torch.stack(target_ids),
        'loss_mask': torch.stack(loss_masks),
        'numerical_targets': torch.stack(targets),
        'latent_features': torch.stack([item['latent_features'] for item in batch]),
        'special_token_positions': torch.stack(special_token_positions),
        'question_texts': [item['question_text'] for item in batch],
        'answer_texts': [item['answer_text'] for item in batch]
    }

def create_sample_data(n_samples: int = 1000, latent_dim: int = 128):
    """Create sample data for testing"""
    # Generate fake latent features
    latent_features = np.random.randn(n_samples, latent_dim)

    # Generate fake metadata
    star_types = ['G-dwarf', 'K-giant', 'M-dwarf', 'F-main', 'A-star', 'B-star']

    metadata = {
        'Teff': np.random.uniform(3000, 10000, n_samples),  # Temperature in K
        'logg': np.random.uniform(1.0, 5.0, n_samples),  # Surface gravity
        'Lstar': np.random.uniform(-2, 2, n_samples),  # Log luminosity
        'type': np.random.choice(star_types, n_samples)
    }

    metadata_df = pd.DataFrame(metadata)
    return latent_features, metadata_df


def setup_dataloaders(
        latent_features: np.ndarray,
        metadata_df: pd.DataFrame,
        llama_tokenizer,
        special_token: str,
        batch_size: int = 8,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        noise_std: float = 0.01,
        random_state: Optional[int] = 42,
        stratify_column: Optional[str] = None,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
        include_latent_analysis: bool = True,
        test: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Setup train, validation, and test dataloaders with proper random splitting.

    Args:
        latent_features: Array of latent feature vectors
        metadata_df: DataFrame containing metadata for each sample
        llama_tokenizer: Tokenizer for text processing
        batch_size: Batch size for all dataloaders
        train_split: Proportion of data for training (default: 0.8)
        val_split: Proportion of data for validation (default: 0.1)
        test_split: Proportion of data for testing (default: 0.1)
        noise_std: Standard deviation of noise to add to training data
        random_state: Random seed for reproducible splits
        stratify_column: Column name in metadata_df to stratify splits by (optional)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for GPU transfer (auto-detected if None)
        include_latent_analysis: Whether to include latent feature analysis in answers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Raises:
        ValueError: If splits don't sum to 1.0 or data lengths don't match
    """

    # Validate inputs
    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split:.3f}")

    if len(latent_features) != len(metadata_df):
        raise ValueError(f"Feature and metadata lengths don't match: {len(latent_features)} vs {len(metadata_df)}")

    if len(latent_features) == 0:
        raise ValueError("Cannot create dataloaders from empty dataset")

    # Auto-detect pin_memory if not specified
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Create indices for splitting
    indices = np.arange(len(latent_features))

    # Prepare stratification if specified
    stratify_data = None
    if stratify_column is not None:
        if stratify_column not in metadata_df.columns:
            raise ValueError(f"Stratify column '{stratify_column}' not found in metadata")
        stratify_data = metadata_df[stratify_column].values

    # First split: separate out test set
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_split,
        random_state=random_state,
        stratify=stratify_data,
        shuffle=True
    )

    # Second split: separate train and validation from remaining data
    # Adjust validation proportion relative to remaining data
    val_size_adjusted = val_split / (train_split + val_split)

    # Prepare stratification for second split if needed
    stratify_train_val = None
    if stratify_data is not None:
        stratify_train_val = stratify_data[train_val_indices]

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_train_val,
        shuffle=True
    )

    # Extract data for each split
    train_features = latent_features[train_indices]
    train_metadata = metadata_df.iloc[train_indices].reset_index(drop=True)

    val_features = latent_features[val_indices]
    val_metadata = metadata_df.iloc[val_indices].reset_index(drop=True)

    test_features = latent_features[test_indices]
    test_metadata = metadata_df.iloc[test_indices].reset_index(drop=True)

    # Print split information
    print(f"Dataset split:")
    print(f"  Train: {len(train_indices)} samples ({len(train_indices) / len(indices) * 100:.1f}%)")
    print(f"  Val:   {len(val_indices)} samples ({len(val_indices) / len(indices) * 100:.1f}%)")
    print(f"  Test:  {len(test_indices)} samples ({len(test_indices) / len(indices) * 100:.1f}%)")

    # Create datasets
    train_dataset = StarDataset(
        train_features,
        train_metadata,
        llama_tokenizer,
        special_token=special_token,
        noise_std=noise_std,
        include_latent_analysis=include_latent_analysis
    )

    val_dataset = StarDataset(
        val_features,
        val_metadata,
        llama_tokenizer,
        special_token=special_token,
        noise_std=0.0,  # No noise for validation
        include_latent_analysis=include_latent_analysis
    )

    test_dataset = StarDataset(
        test_features,
        test_metadata,
        llama_tokenizer,
        special_token=special_token,
        noise_std=0.0,  # No noise for testing
        include_latent_analysis=include_latent_analysis
    )

    if test:
        test_dataset_samples(train_dataset, 100)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Keep all validation samples
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Keep all test samples
    )

    if test:
        for i in range(10):
            start = time.time()
            data = next(iter(test_loader))
            print("dataloader time: ", time.time() - start)


    return train_loader, val_loader, test_loader


def test_dataset_samples(dataset, num_samples):
    for i in range(num_samples):
        data = dataset[i]
        print(data['numerical_targets'])


if __name__ == "__main__":
    # Example usage
    print("Creating sample data...")
    latent_features, metadata_df = create_sample_data(n_samples=100, latent_dim=128)

    print(f"Created {len(latent_features)} samples")
    print(f"Latent features shape: {latent_features.shape}")
    print(f"Metadata columns: {list(metadata_df.columns)}")
    print(f"Sample metadata:\n{metadata_df.head()}")

    # Example of enhanced dataset with latent analysis
    print("\n=== Testing Enhanced Dataset ===")
