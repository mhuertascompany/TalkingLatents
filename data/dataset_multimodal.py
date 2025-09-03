import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict, Any, List
import os
from pathlib import Path
from astropy.io import fits

from llama3.llama.tokenizer import Tokenizer
from data.transforms import RandomMasking




class StellarDataset(Dataset):
    """
    PyTorch Dataset for stellar descriptions and optional spectral features
    Now includes tokenization for LLaMA
    
    Args:
        json_file (str): Path to the JSON file with stellar data
        features_array (Optional[np.ndarray]): Optional array of spectral features
        split (str): One of 'train', 'val', 'test'
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set  
        test_ratio (float): Proportion for test set (remaining after train/val)
        random_state (int): Random seed for reproducible splits
        filter_valid_descriptions (bool): Whether to filter out samples with no description
        cache_dir (Optional[str]): Directory to cache split indices for consistency
        tokenizer_path (Optional[str]): Path to SentencePiece tokenizer model
        max_length (int): Maximum sequence length for tokenization
    """
    
    def __init__(self, 
                 json_file: str,
                 features_array: Optional[np.ndarray] = None,
                 split: str = 'train',
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42,
                 spectral_transforms: Optional[Any] = None,
                 filter_valid_descriptions: bool = True,
                 cache_dir: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 max_length: int = 512):
        
        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', got {split}"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.json_file = json_file
        self.features_array = features_array
        self.split = split
        self.random_state = random_state
        self.filter_valid_descriptions = filter_valid_descriptions
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.tokenizer = None
        self.transforms = spectral_transforms
        self.mask_transform = RandomMasking()  # Example masking
        
        # Load tokenizer if available
        self._load_tokenizer()
        
        # Load and process data
        self._load_data()
        self._create_splits(train_ratio, val_ratio, test_ratio, cache_dir)
        
    def _load_tokenizer(self):
        """Load SentencePiece tokenizer if available"""
        if self.tokenizer_path and os.path.exists(self.tokenizer_path):
            try:
                self.tokenizer = Tokenizer(model_path=self.tokenizer_path)
                print(f"Loaded tokenizer from {self.tokenizer_path}")
            except ImportError:
                print("SentencePiece not available. Install with: pip install sentencepiece")
                self.tokenizer = None
            except Exception as e:
                print(f"Failed to load tokenizer: {e}")
                self.tokenizer = None
        else:
            print(f"Tokenizer path not found: {self.tokenizer_path}")
            self.tokenizer = None
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text using tokenizer or fallback"""
        if self.tokenizer is not None:
            # Use real tokenizer
            token_ids = self.tokenizer.encode(text, bos=True, eos=False)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                # Pad with pad token (usually 0)
                pad_token_id = self.tokenizer.pad_id if hasattr(self.tokenizer, 'pad_id') else 0
                token_ids = token_ids + [pad_token_id] * (self.max_length - len(token_ids))
            return torch.tensor(token_ids, dtype=torch.long)
        else:
            print("Tokenizer not available, using fallback tokenization.")
            # Fallback: create deterministic tokens based on text
            # Simple hash-based tokenization for consistent results
            words = text.lower().split()
            token_ids = []
            
            for word in words:
                # Create consistent token ID from word hash
                word_hash = hash(word) % 10000  # Limit vocab size
                token_ids.append(abs(word_hash) + 1)  # Avoid 0 (pad token)
                
                if len(token_ids) >= self.max_length:
                    break
            
            # Pad to max_length
            while len(token_ids) < self.max_length:
                token_ids.append(0)  # Pad token
                
            return torch.tensor(token_ids[:self.max_length], dtype=torch.long)
        
    def _load_data(self):
        """Load data from JSON file"""
        print(f"Loading data from {self.json_file}...")
        
        with open(self.json_file, 'r') as f:
            self.raw_data = json.load(f)
            
        print(f"Loaded {len(self.raw_data)} samples from JSON")
        
        # Filter samples with valid descriptions if requested
        if self.filter_valid_descriptions:
            valid_samples = []
            for sample in self.raw_data:
                desc = sample.get('description')
                if desc and desc.strip() and desc.lower() not in ['none', 'null', '']:
                    valid_samples.append(sample)
            
            print(f"Filtered to {len(valid_samples)} samples with valid descriptions")
            self.raw_data = valid_samples
        
        # Extract dataframe indices
        self.df_indices = []
        for sample in self.raw_data:
            df_idx = sample.get('index')
            if df_idx is not None:
                self.df_indices.append(df_idx)
            else:
                print(f"Warning: Sample missing dataframe index: {sample.get('obsid', 'unknown')}")
                
        # Validate features array if provided
        if self.features_array is not None:
            max_idx = max(self.df_indices) if self.df_indices else -1
            if max_idx >= len(self.features_array):
                raise ValueError(f"Features array length ({len(self.features_array)}) is smaller than "
                               f"maximum dataframe index ({max_idx})")
            print(f"Features array validated. Shape: {self.features_array.shape}")
            
    def _create_splits(self, train_ratio: float, val_ratio: float, test_ratio: float, cache_dir: Optional[str] = None):
        """Create train/val/test splits with caching for consistency"""
        
        n_samples = len(self.raw_data)
        indices = np.arange(n_samples)
        
        # Create cache key based on file and parameters
        cache_key = f"{Path(self.json_file).stem}_{n_samples}_{train_ratio}_{val_ratio}_{test_ratio}_{self.random_state}"
        cache_file = None
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"splits_{cache_key}.npz")
        
        # Try to load cached splits
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached splits from {cache_file}")
            cached = np.load(cache_file)
            train_indices = cached['train_indices']
            val_indices = cached['val_indices'] 
            test_indices = cached['test_indices']
        else:
            print("Creating new train/val/test splits...")
            
            # First split: separate test set
            temp_indices, test_indices = train_test_split(
                indices, 
                test_size=test_ratio,
                random_state=self.random_state,
                shuffle=True
            )
            
            # Second split: separate train and val from remaining data
            adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
            train_indices, val_indices = train_test_split(
                temp_indices,
                test_size=adjusted_val_ratio,
                random_state=self.random_state,
                shuffle=True
            )
            
            # Cache splits if directory provided
            if cache_file:
                print(f"Caching splits to {cache_file}")
                np.savez(cache_file, 
                        train_indices=train_indices,
                        val_indices=val_indices, 
                        test_indices=test_indices)
        
        # Select indices for current split
        if self.split == 'train':
            self.split_indices = train_indices
        elif self.split == 'val':
            self.split_indices = val_indices
        else:  # test
            self.split_indices = test_indices
            
        print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        print(f"Current split ({self.split}): {len(self.split_indices)} samples")
        
    def read_lamost_spectra(self, filename):
        try:
            with fits.open(filename) as hdulist:
                binaryext = hdulist[1].data
                header = hdulist[0].header
            spectra = torch.tensor(binaryext['FLUX'].astype(np.float32))
            wv = binaryext['WAVELENGTH'].astype(np.float32)
            rv = header['HELIO_RV']
            meta = {'RV': rv, 'wavelength': wv}
        except FileNotFoundError:
            print(f"File not found: {filename}")
            spectra = torch.zeros(1, 4096, dtype=torch.float32)
            wv = np.linspace(3690, 9100, 4096, dtype=np.float32)
            meta = {'RV': 0.0, 'wavelength': wv}
        if self.transforms:
            spectra, _, meta = self.transforms(spectra, None, meta)
        spectra_masked, mask, _ = self.mask_transform(spectra, None, meta)
        pad = torch.zeros(1, 4096 - spectra.shape[-1])
        spectra = torch.cat([spectra, pad], dim=-1)
        spectra_masked = torch.cat([spectra_masked, pad], dim=-1)
        return spectra, spectra_masked, meta
    
    def read_apogee_spectra(self, filename):
        try:
            with fits.open(filename) as hdul:
                spectra = torch.tensor(hdul[1].data.astype(np.float32).squeeze()[None])
            meta = {}
            header = hdul[1].header
            # Create pixel array (1-indexed for FITS convention)
            pixels = np.arange(1, spectra.shape[-1] + 1)
            
            # Calculate log10(wavelength):
            # log_wave = CRVAL1 + CDELT1 * (pixel - CRPIX1)
            log_wavelength = header['CRVAL1'] + header['CDELT1'] * (pixels - header['CRPIX1'])
            
            # Convert to linear wavelength in Angstroms
            wv = 10**log_wavelength
            meta = {'wavelength': wv}
        except FileNotFoundError:
            print(f"File not found: {filename}")
            spectra = torch.zeros(1, 8576, dtype=torch.float32)
            wv = np.linspace(15100, 17000, 8576, dtype=np.float32)
            meta = {'wavelength': wv}
        if self.transforms:
            spectra, _, meta = self.transforms(spectra, None, meta)
        spectra_masked, mask, _ = self.mask_transform(spectra, None, meta)
        pad = torch.zeros(1, 8576 - spectra.shape[-1])
        spectra = torch.cat([spectra, pad], dim=-1)
        spectra_masked = torch.cat([spectra_masked, pad], dim=-1)
        return spectra, spectra_masked, meta
    
    def get_raw_spectra(self, obsid: int, id_type='obsid') -> Optional[np.ndarray]:
        if id_type == 'obsid':
                obsdir = str(obsid)[:4]
                spectra_filename = os.path.join(f'/data/lamost/data', f'{obsdir}/{obsid}.fits')
                spectra, spectra_masked, meta = self.read_lamost_spectra(spectra_filename)
                meta['obsid'] = obsid
        elif id_type == 'APOGEE_ID':
            spectra_filename = f"/data/apogee/data/aspcapStar-dr17-{obsid}.fits"
            spectra, spectra_masked, meta = self.read_apogee_spectra(spectra_filename)
            meta['apogee_id'] = obsid
        else:
            raise ValueError(f"Unknown obsid format: {id_type}")
        return spectra, spectra_masked, meta
    
    def __len__(self):
        return len(self.split_indices)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample with tokenized description
        
        Returns:
            dict: Contains 'description_tokens', 'description_text', 'features', 'stellar_data', 'obsid', 'df_index'
        """
        # Get the actual sample index
        sample_idx = self.split_indices[idx]
        sample = self.raw_data[sample_idx]
        
        # Extract description and tokenize
        description_text = sample.get('description', '')
        description_tokens = self._tokenize_text(description_text)
        
        # Extract features if available
        df_index = sample.get('index')
        spectra, masked_spectra, _ = self.get_raw_spectra(sample['obsid'])
        if self.features_array is not None and df_index is not None:
            features = torch.tensor(self.features_array[df_index].astype(np.float32))
        else:
            features = masked_spectra  # Use raw spectra if no features array provided
        
        # Extract other useful information
        stellar_data = sample.get('stellar_data', {})
        obsid = sample.get('obsid', None)
        
        return {
            'description_tokens': description_tokens,  # Tokenized for LLaMA
            'description_text': description_text,      # Original text
            'features': features,
            'spectra': spectra,
            'masked_spectra': masked_spectra,
            'stellar_data': stellar_data,
            'obsid': obsid,
            'df_index': df_index,
            'sample_index': sample_idx
        }
    
    def get_feature_dim(self) -> Optional[int]:
        """Get the dimensionality of features"""
        if self.features_array is not None:
            return self.features_array.shape[1] if len(self.features_array.shape) > 1 else 1
        return None
    
    def get_split_info(self) -> Dict[str, int]:
        """Get information about all splits"""
        # This requires recreating splits temporarily
        indices = np.arange(len(self.raw_data))
        temp_indices, test_indices = train_test_split(
            indices, test_size=0.15, random_state=self.random_state, shuffle=True
        )
        train_indices, val_indices = train_test_split(
            temp_indices, test_size=0.15/(0.7+0.15), random_state=self.random_state, shuffle=True
        )
        
        return {
            'total': len(self.raw_data),
            'train': len(train_indices),
            'val': len(val_indices),
            'test': len(test_indices)
        }


def create_stellar_dataloaders(json_file: str,
                             features_array: Optional[np.ndarray] = None,
                             batch_size: int = 32,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             random_state: int = 42,
                             num_workers: int = 0,
                             cache_dir: Optional[str] = None,
                             **dataset_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train, val, test dataloaders
    """
    
    # Create datasets for each split
    train_dataset = StellarDataset(
        json_file=json_file,
        features_array=features_array,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        **dataset_kwargs
    )
    
    val_dataset = StellarDataset(
        json_file=json_file,
        features_array=features_array,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        **dataset_kwargs
    )
    
    test_dataset = StellarDataset(
        json_file=json_file,
        features_array=features_array,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        cache_dir=cache_dir,
        **dataset_kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle tokenized descriptions and optional features
    """
    description_tokens = [item['description_tokens'] for item in batch]
    description_texts = [item['description_text'] for item in batch]
    obsids = [item['obsid'] for item in batch]
    df_indices = [item['df_index'] for item in batch]
    stellar_data = [item['stellar_data'] for item in batch]
    spectra = [item['spectra'] for item in batch]
    masked_spectra = [item['masked_spectra'] for item in batch]
    
    # Stack tokenized descriptions
    description_tokens_tensor = torch.stack(description_tokens)  # [batch_size, seq_len]
    
    # Handle features - check if any sample has features
    features_list = [item['features'] for item in batch]
    if any(f is not None for f in features_list):
        # Stack features, replacing None with zeros
        feature_dim = None
        for f in features_list:
            if f is not None:
                feature_dim = f.shape[0] if len(f.shape) == 1 else f.shape
                break
        
        if feature_dim is not None:
            processed_features = []
            for f in features_list:
                if f is not None:
                    processed_features.append(f)
                else:
                    processed_features.append(torch.zeros(feature_dim, dtype=torch.float32))
            features_tensor = torch.stack(processed_features)
        else:
            features_tensor = None
    else:
        features_tensor = None
    
    return {
        'description_tokens': description_tokens_tensor,  # For LLaMA model
        'descriptions': description_texts,                # For logging/debugging
        'features': features_tensor,
        'spectra': torch.stack(spectra),
        'masked_spectra': torch.stack(masked_spectra),
        'obsids': obsids,
        'df_indices': df_indices,
        'stellar_data': stellar_data
    }


# Example usage and testing
if __name__ == "__main__":

    import os
    os.system('pip install tiktoken fairscale fire blobfile')
    import sys
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(ROOT_DIR)
    print("running from ", ROOT_DIR) 

    json_path = '/data/TalkingLatents/data/dataset/stellar_descriptions.json'
    spectral_features = np.load('/data/TalkingLatents/logs/2025-07-29/features.npy')
    # Example usage
    print("Example usage:")
    
    # Case 1: JSON only (no features)
    print("\n1. Dataset with descriptions only:")
    
    dataset = StellarDataset(
        json_file=json_path,
        split='train'
    )
    
    # Case 2: JSON + features array
    print("\n2. Dataset with descriptions and spectral features:")
       
    # Create dataset
    dataset = StellarDataset(
        json_file=json_path,
        features_array=spectral_features,
        split='train'
    )
    
    # Case 3: Create all dataloaders at once
    print("\n3. Create train/val/test dataloaders:")
    train_loader, val_loader, test_loader = create_stellar_dataloaders(
        json_file=json_path,
        features_array=spectral_features,  # Optional
        batch_size=32,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        cache_dir='cache/'  # Cache splits for consistency
    )
    
    # Use custom collate function if needed
    train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    data = next(iter(train_loader))
    print(f"Batch description tokens shape: {data['description_tokens'].shape}")
    
    # Case 4: Iterate through dataset
    print("\n4. Iterate through dataset:")
    for batch in train_loader:
        descriptions = batch['descriptions']  # List of strings
        features = batch['features']          # Tensor [batch_size, feature_dim] or None
        obsids = batch['obsids']             # List of observation IDs
        stellar_data = batch['stellar_data'] # List of stellar parameter dicts
        
        # Your training code here
        break
    
    # Test with dummy data if files exist
    try:
        if os.path.exists(json_path):
            print("\n=== Testing with actual data ===")
            
            # Test without features
            dataset = StellarDataset(
                json_file=json_path,
                split='train'
            )
            print(f"Dataset length: {len(dataset)}")
            print(f"Feature dimension: {dataset.get_feature_dim()}")
            
            # Test one sample
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Description preview: {sample['description'][:100]}...")
            
    except Exception as e:
        print(f"Could not test with actual data: {e}")
    
    print("\n=== Dataset Features ===")
    print("✅ Automatic train/val/test splitting with consistent random seeds")
    print("✅ Optional spectral features support")  
    print("✅ Caching of split indices for reproducibility")
    print("✅ Filtering of samples with invalid descriptions")
    print("✅ Custom collate function for handling mixed data types")
    print("✅ Proper dataframe index mapping for features")
    print("✅ GPU memory pinning support")