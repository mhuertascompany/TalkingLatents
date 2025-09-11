import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import os
from typing import Optional

# Import your existing data components
from data.dataset_interpert import StellarQuestionsDataset, create_stellar_dataloaders
from data.transforms import Compose, GeneralSpectrumPreprocessor, ToTensor


class StellarDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for stellar data
    """
    
    def __init__(
        self,
        json_file: str,
        tokenizer_path: str,
        features_file: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        max_seq_length: int = 128,
        num_spectral_features: int = 1,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        self.json_file = json_file
        self.tokenizer_path = tokenizer_path
        self.features_file = features_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.max_seq_length = max_seq_length
        self.num_spectral_features = num_spectral_features
        self.cache_dir = cache_dir or 'cache'
        
        # Create transforms
        self.transforms = Compose([
            GeneralSpectrumPreprocessor(rv_norm=True), 
            ToTensor()
        ])
        
        # Placeholders
        self.spectral_features = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """
        Called only on 1 GPU/TPU in distributed mode
        Use this for downloading/preparing data that shouldn't be done in parallel
        """
        # Load spectral features if provided
        if self.features_file and os.path.exists(self.features_file):
            print(f"Loading spectral features from {self.features_file}")
            self.spectral_features = np.load(self.features_file)
            print(f"Spectral features shape: {self.spectral_features.shape}")
        else:
            print("No spectral features file provided. Will use raw spectra on-the-fly.")
            self.spectral_features = None
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def setup(self, stage: Optional[str] = None):
        """
        Called on every GPU in distributed mode
        Split datasets, apply transforms, etc.
        """
        
        # Create datasets for each split
        if stage == "fit" or stage is None:
            self.train_dataset = StellarQuestionsDataset(
                json_file=self.json_file,
                features_array=self.spectral_features,
                spectral_transforms=self.transforms,
                split='train',
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                random_state=self.random_seed,
                cache_dir=self.cache_dir,
                tokenizer_path=self.tokenizer_path,
                max_length=self.max_seq_length,
                num_spectral_features=self.num_spectral_features,
            )
            
            self.val_dataset = StellarQuestionsDataset(
                json_file=self.json_file,
                features_array=self.spectral_features,
                spectral_transforms=self.transforms,
                split='val',
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                random_state=self.random_seed,
                cache_dir=self.cache_dir,
                tokenizer_path=self.tokenizer_path,
                max_length=self.max_seq_length,
                num_spectral_features=self.num_spectral_features,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = StellarQuestionsDataset(
                json_file=self.json_file,
                features_array=self.spectral_features,
                spectral_transforms=self.transforms,
                split='test',
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                random_state=self.random_seed,
                cache_dir=self.cache_dir,
                tokenizer_path=self.tokenizer_path,
                max_length=self.max_seq_length,
                num_spectral_features=self.num_spectral_features,
            )
        
        # Print dataset info
        if stage == "fit" or stage is None:
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            print(f"Test dataset size: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        """Return training dataloader"""
        from data.dataset_interpert import collate_fn
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        from data.dataset_interpert import collate_fn
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self):
        """Return test dataloader"""
        from data.dataset_interpert import collate_fn
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )
    
    def predict_dataloader(self):
        """Return prediction dataloader (same as test)"""
        return self.test_dataloader()
    
    def teardown(self, stage: Optional[str] = None):
        """Called at the end of fit, validate, test, predict, or teardown"""
        # Clean up any resources if needed
        pass