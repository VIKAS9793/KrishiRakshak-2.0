"""PyTorch Lightning DataModule for KrishiRakshak."""
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image

from src.config import Config


class PlantDiseaseDataset(Dataset):
    """PyTorch Dataset for plant disease classification."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
        is_test: bool = False
    ):
        """
        Args:
            df: DataFrame with 'image_path' and 'label' columns
            transform: Albumentations transform
            class_to_idx: Mapping from class names to indices
            is_test: If True, returns (image, ) instead of (image, label)
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        
        # Create or use provided class mapping
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = class_to_idx or {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Cache for images
        self.image_paths = self.df['image_path'].values
        self.labels = None if is_test else [self.class_to_idx[cls] for cls in self.df['label']]
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        img_path = self.image_paths[idx]
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            
            # Apply transforms
            if self.transform:
                augmented = self.transform(image=img)
                img = augmented['image']
            
            if self.is_test:
                return img
                
            label = self.labels[idx]
            return img, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random valid example if there's an error
            return self[np.random.randint(0, len(self) - 1)]


class PlantDiseaseDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for plant disease classification."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.class_weights = None
        self.class_to_idx = None
        
    def prepare_data(self):
        """Download data if needed. Not needed for local files."""
        pass
        
    def setup(self, stage: Optional[str] = None):
        """Load and split data."""
        # Load data
        train_df = pd.read_csv(self.config.TRAIN_CSV)
        val_df = pd.read_csv(self.config.VAL_CSV)
        test_df = pd.read_csv(self.config.TEST_CSV)
        
        # Load class weights
        with open(self.config.CLASS_WEIGHTS_JSON) as f:
            self.class_weights = json.load(f)
        
        # Create class mapping from training data
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(train_df['label'].unique()))}
        
        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_ds = PlantDiseaseDataset(
                train_df,
                transform=self.config.get_train_transforms(),
                class_to_idx=self.class_to_idx
            )
            self.val_ds = PlantDiseaseDataset(
                val_df,
                transform=self.config.get_val_test_transforms(),
                class_to_idx=self.class_to_idx
            )
            
        if stage == 'test' or stage is None:
            self.test_ds = PlantDiseaseDataset(
                test_df,
                transform=self.config.get_val_test_transforms(),
                class_to_idx=self.class_to_idx
            )
    
    def train_dataloader(self):
        """Create training DataLoader with class balancing."""
        # Calculate sample weights for imbalanced classes
        labels = self.train_ds.labels
        class_weights = [self.class_weights[self.train_ds.idx_to_class[cls]] for cls in labels]
        sampler = WeightedRandomSampler(
            weights=class_weights,
            num_samples=len(labels),
            replacement=True
        )
        
        return DataLoader(
            self.train_ds,
            batch_size=self.config.BATCH_SIZE,
            sampler=sampler,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )
    
    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx) if self.class_to_idx else 0
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return list(self.class_to_idx.keys()) if self.class_to_idx else []
