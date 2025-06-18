"""PyTorch Lightning DataModule for KrishiRakshak.

This module handles data loading, preprocessing, and augmentation for the plant disease
classification task. It supports:
- Efficient data loading with caching
- Advanced data augmentation
- Class imbalance handling
- Test-time augmentation (TTA)
- Distributed training support
"""
import json
import logging
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile, UnidentifiedImageError
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import Config

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PlantDiseaseDataset(Dataset):
    """PyTorch Dataset for plant disease classification with advanced features.
    
    Features:
    - Handles corrupted images with automatic fallback
    - Image caching for faster training
    - Support for both training and inference modes
    - Comprehensive error handling and logging
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
        is_test: bool = False,
        cache_images: bool = False
    ) -> None:
        """Initialize the dataset.
        
        Args:
            df: DataFrame containing 'image_path' and 'label' columns
            transform: Albumentations transform pipeline. If None, a minimal transform will be used.
            class_to_idx: Optional mapping from class names to indices
            is_test: If True, returns only images without labels
            cache_images: If True, cache images in memory for faster training
        """
        self.df = df.reset_index(drop=True)
        
        # Ensure we always have a transform
        if transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform
        self.is_test = is_test
        self.cache_images = cache_images
        self.cache = {}
        
        # Validate dataframe
        self._validate_dataframe()
        
        # Create or use provided class mapping
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = class_to_idx or {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Store paths and labels
        self.image_paths = self.df['image_path'].values
        self.labels = None if is_test else [self.class_to_idx[cls] for cls in self.df['label']]
        
        # Track problematic indices
        self.bad_indices = set()
        
        # Initialize cache if enabled
        if self.cache_images:
            self._preload_images()
    
    def _validate_dataframe(self) -> None:
        """Validate the input dataframe structure."""
        required_columns = {'image_path', 'label'}
        if not required_columns.issubset(self.df.columns):
            missing = required_columns - set(self.df.columns)
            raise ValueError(f"Missing required columns in DataFrame: {missing}")
    
    def _preload_images(self):
        """Preload all images into memory if cache is enabled."""
        from tqdm import tqdm
        
        print("Preloading images into memory...")
        for idx in tqdm(range(len(self.image_paths))):
            if idx not in self.bad_indices:
                self._load_image(idx)
    
    def _load_image(self, idx: int) -> Optional[np.ndarray]:
        """Load and validate an image.
        
        Args:
            idx: Index of the image to load
            
        Returns:
            Loaded image as numpy array or None if loading failed
        """
        if idx in self.cache:
            return self.cache[idx]
            
        img_path = self.image_paths[idx]
        
        try:
            # Try to load the image
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_array = np.array(img)
                
                # Basic validation
                if img_array.size == 0:
                    raise ValueError(f"Empty image: {img_path}")
                
                # Cache if enabled
                if self.cache_images:
                    self.cache[idx] = img_array
                    
                return img_array
                
        except (IOError, ValueError, UnidentifiedImageError, OSError) as e:
            self.bad_indices.add(idx)
            if len(self.bad_indices) < 10:  # Only log first 10 errors to avoid spam
                logger.warning(f"Could not load image {img_path}: {str(e)}")
            return None
    
    def __len__(self) -> int:
        """Return the number of valid items in the dataset."""
        return len(self.df) - len(self.bad_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get an item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple containing (image_tensor, ) for test or (image_tensor, label) for train/val
            
        Note:
            If an image fails to load, this will try to return a random valid sample
        """
        max_attempts = 3
        for _ in range(max_attempts):
            try:
                # Skip known bad indices
                while idx in self.bad_indices:
                    idx = np.random.randint(0, len(self.image_paths))
                
                # Load image
                img_array = self._load_image(idx)
                if img_array is None:
                    raise ValueError(f"Failed to load image at index {idx}")
                
                # Apply transforms - always use a transform
                augmented = self.transform(image=img_array)
                img_tensor = augmented['image']
                
                # Return based on mode
                if self.is_test:
                    return (img_tensor,)
                    
                label = self.labels[idx]
                return img_tensor, torch.tensor(label, dtype=torch.long)
                
            except Exception as e:
                if idx not in self.bad_indices:
                    self.bad_indices.add(idx)
                    if len(self.bad_indices) < 10:  # Don't spam logs
                        print(f"Error processing item {idx}: {str(e)}")
                
                # Try a different random index
                idx = np.random.randint(0, len(self.image_paths))
        
        # If we get here, all attempts failed
        raise RuntimeError(f"Failed to load any valid sample after {max_attempts} attempts")


class PlantDiseaseDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for plant disease classification.
    
    Handles data loading, splitting, and augmentation for training, validation, and testing.
    Provides functionality for:
    - Automatic train/val/test split with stratification
    - Class balancing with weighted sampling
    - Image caching for faster training
    - Comprehensive data augmentation
    - Test-time augmentation support
    """
    
    def __init__(
        self,
        config: Config,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        persistent_workers: Optional[bool] = None,
        pin_memory: Optional[bool] = None,
        drop_last: bool = False,
        shuffle: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize the data module with configuration.
        
        Args:
            config: Configuration object
            batch_size: Override config batch size if needed
            num_workers: Override config num_workers if needed
            train_transform: Transformations for training data
            val_transform: Transformations for validation data
            test_transform: Transformations for test data
            persistent_workers: Override config persistent_workers if needed
            pin_memory: Override config pin_memory if needed
            drop_last: If True, drops the last incomplete batch
            shuffle: If True, shuffles training data
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.config = config
        
        # Use provided values or fall back to config
        self.batch_size = batch_size or config.BATCH_SIZE
        self.num_workers = num_workers or config.NUM_WORKERS
        self.persistent_workers = (persistent_workers 
                                 if persistent_workers is not None 
                                 else config.PERSISTENT_WORKERS)
        self.pin_memory = (pin_memory 
                          if pin_memory is not None 
                          else config.PIN_MEMORY)
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Data splits
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        # Class information
        self.class_weights = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.num_classes = 0
        
        # Transforms - use provided or get from config
        self.train_transform = train_transform or config.get_train_transforms()
        self.val_transform = val_transform or config.get_val_transforms()
        self.test_transform = test_transform or config.get_test_transforms()
        
        # Initialize datasets
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        
        # Set random seed for reproducibility
        pl.seed_everything(config.SEED)
        
    def prepare_data(self):
        """Download or verify data files.
        
        This method is called only once across all GPUs in distributed training.
        """
        # Verify data directory exists
        if not os.path.exists(self.config.DATA_DIR):
            raise FileNotFoundError(f"Data directory not found: {self.config.DATA_DIR}")
            
        # Create required directories
        os.makedirs(self.config.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.config.REPORTS_DIR, exist_ok=True)
        
    def _load_dataframe(self) -> pd.DataFrame:
        """Load and validate the dataset from the data directory.
        
        Returns:
            DataFrame containing image paths and labels
            
        Raises:
            ValueError: If no valid images are found or insufficient classes
        """
        all_data = []
        
        # Scan data directory for images
        for class_name in os.listdir(self.config.DATA_DIR):
            class_dir = os.path.join(self.config.DATA_DIR, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_data.append({
                            'image_path': os.path.join(class_dir, img_name),
                            'label': class_name
                        })
        
        if not all_data:
            raise ValueError(f"No images found in {self.config.DATA_DIR}")
        
        # Create DataFrame and validate
        df = pd.DataFrame(all_data)
        
        # Ensure we have enough samples per class
        class_counts = df['label'].value_counts()
        valid_classes = class_counts[class_counts >= self.config.MIN_SAMPLES_PER_CLASS].index
        
        if len(valid_classes) < 2:
            raise ValueError(f"Need at least 2 classes with {self.config.MIN_SAMPLES_PER_CLASS} samples each. "
                           f"Found classes: {class_counts.to_dict()}")
        
        # Filter to only include valid classes
        df = df[df['label'].isin(valid_classes)].reset_index(drop=True)
        return df
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame with all data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: training and temp (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=0.3,  # 30% for validation + test
            random_state=self.config.SEED,
            stratify=df['label']
        )
        
        # Second split: validation and test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,  # 15% each for val and test
            random_state=self.config.SEED,
            stratify=temp_df['label']
        )
        
        return train_df.reset_index(drop=True), \
               val_df.reset_index(drop=True), \
               test_df.reset_index(drop=True)
    
    def setup(self, stage: Optional[str] = None):
        """Set up data for training, validation, and testing.
        
        Args:
            stage: Either 'fit', 'test', or None for both
        """
        if stage == 'fit' or stage is None:
            # Load and split data
            df = self._load_dataframe()
            self.train_df, self.val_df, self.test_df = self._split_data(df)
            
            # Calculate class weights for imbalanced data
            self._calculate_class_weights()
            
            # Create class to index mapping
            self.class_to_idx = {cls: i for i, cls in enumerate(sorted(df['label'].unique()))}
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            
            # Log dataset statistics
            self._log_dataset_stats()
            
            # Initialize datasets
            self.train_ds = PlantDiseaseDataset(
                df=self.train_df,
                transform=self.config.get_train_transforms(),
                class_to_idx=self.class_to_idx,
                cache_images=self.config.CACHE_IMAGES
            )
            
            self.val_ds = PlantDiseaseDataset(
                df=self.val_df,
                transform=self.config.get_val_transforms(),
                class_to_idx=self.class_to_idx,
                cache_images=self.config.CACHE_IMAGES
            )
        
        if stage == 'test' or stage is None:
            if self.test_df is None:
                df = self._load_dataframe()
                _, _, self.test_df = self._split_data(df)
            
            self.test_ds = PlantDiseaseDataset(
                df=self.test_df,
                transform=self.config.get_test_transforms(),
                class_to_idx=self.class_to_idx,
                is_test=True,
                cache_images=self.config.CACHE_IMAGES
            )
    
    def _worker_init_fn(self, worker_id: int) -> None:
        """Worker initialization function for reproducibility.
        
        Args:
            worker_id: Worker process ID
        """
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        np.random.seed(worker_id)
        random.seed(worker_id)
        torch.manual_seed(worker_seed)
    
    def get_train_transforms(self) -> A.Compose:
        """Create training data augmentation pipeline.
        
        Returns:
            A.Compose: Albumentations composition of transforms
        """
        transforms = [
            A.HorizontalFlip(p=self.config.HORIZONTAL_FLIP_PROB),
            A.VerticalFlip(p=self.config.VERTICAL_FLIP_PROB),
            A.Rotate(limit=self.config.ROTATION_RANGE, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=self.config.SHIFT_LIMIT,
                scale_limit=0,
                rotate_limit=0,
                p=0.5
            ),
            A.RandomResizedCrop(
                height=self.config.IMG_SIZE,
                width=self.config.IMG_SIZE,
                scale=self.config.ZOOM_RANGE,
                ratio=(0.8, 1.2),
                p=0.8
            ),
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=self.config.BRIGHTNESS_LIMIT,
                        contrast=self.config.CONTRAST_LIMIT,
                        saturation=self.config.SATURATION_LIMIT,
                        hue=self.config.HUE_LIMIT,
                        p=0.8
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.8
                    ),
                ],
                p=0.9,
            ),
            A.Normalize(
                mean=self.config.MEAN,
                std=self.config.STD,
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()
        ]
        
        # Add Cutout if enabled
        if getattr(self.config, 'CUTOUT_ENABLED', False):
            transforms.insert(
                -2,  # Before normalization
                A.CoarseDropout(
                    max_holes=self.config.CUTOUT_NUM_HOLES,
                    max_height=self.config.CUTOUT_MAX_H_SIZE,
                    max_width=self.config.CUTOUT_MAX_W_SIZE,
                    fill_value=self.config.CUTOUT_FILL_VALUE,
                    p=0.5
                )
            )
        
        # Add RandomErasing if enabled
        if getattr(self.config, 'USE_RANDOM_ERASING', False):
            transforms.append(
                A.RandomErasing(
                    scale=self.config.RANDOM_ERASING_SCALE,
                    ratio=self.config.RANDOM_ERASING_RATIO,
                    value='random',
                    p=self.config.RANDOM_ERASING_PROB
                )
            )
            
        return A.Compose(transforms, p=self.config.AUGMENTATION_PROB)
    
    def get_val_transforms(self) -> A.Compose:
        """Create validation data transforms.
        
        Returns:
            A.Compose: Albumentations composition of transforms
        """
        return A.Compose([
            A.Resize(self.config.IMG_SIZE, self.config.IMG_SIZE),
            A.Normalize(
                mean=self.config.MEAN,
                std=self.config.STD,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        
    def get_test_transforms(self) -> Union[A.Compose, List[A.Compose]]:
        """Create test data transforms with optional TTA.
        
        Returns:
            Union[A.Compose, List[A.Compose]]: Single transform or list of transforms for TTA
        """
        base_transform = [
            A.Resize(self.config.IMG_SIZE, self.config.IMG_SIZE),
            A.Normalize(
                mean=self.config.MEAN,
                std=self.config.STD,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
        
        if not getattr(self.config, 'TTA_NUM_AUGS', 0):
            return A.Compose(base_transform)
            
        # Create TTA transforms
        tta_transforms = []
        for _ in range(self.config.TTA_NUM_AUGS):
            tta_transforms.append(A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=self.config.ROTATION_RANGE, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                *base_transform
            ]))
            
        return tta_transforms
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader with class balancing.
        
        Returns:
            DataLoader: Configured DataLoader for training data
        """
        if self.train_ds is None:
            self.setup(stage='fit')
            
        # Calculate sample weights for imbalanced classes
        labels = self.train_ds.labels
        class_weights = [self.class_weights[self.train_ds.idx_to_class[cls]] 
                        for cls in labels]
        
        sampler = WeightedRandomSampler(
            weights=class_weights,
            num_samples=len(labels),
            replacement=True
        )
        
        # Get base DataLoader arguments
        kwargs = self._get_dataloader_kwargs()
        kwargs.update({
            'dataset': self.train_ds,
            'sampler': sampler,
            'drop_last': self.drop_last
        })
        
        return DataLoader(**kwargs)
    
    def _get_dataloader_kwargs(self) -> dict:
        """Get common DataLoader arguments from config.
        
        Returns:
            dict: Dictionary of DataLoader arguments
        """
        kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers and (self.num_workers > 0),
            'worker_init_fn': self._worker_init_fn,
            'prefetch_factor': getattr(self.config, 'PREFETCH_FACTOR', 2),
            'timeout': getattr(self.config, 'DATALOADER_TIMEOUT', 60)
        }
        
        # Add multiprocessing context if specified
        if hasattr(self.config, 'MULTIPROCESSING_CONTEXT') and self.config.MULTIPROCESSING_CONTEXT:
            import multiprocessing as mp
            kwargs['multiprocessing_context'] = mp.get_context(self.config.MULTIPROCESSING_CONTEXT)
            
        return kwargs
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader.
        
        Returns:
            DataLoader: Configured DataLoader for validation data
        """
        if self.val_ds is None:
            self.setup(stage='fit')
            
        kwargs = self._get_dataloader_kwargs()
        kwargs.update({
            'dataset': self.val_ds,
            'shuffle': False,
            'drop_last': False
        })
        
        return DataLoader(**kwargs)
        
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Create test DataLoader with optional TTA.
        
        Returns:
            Union[DataLoader, List[DataLoader]]: 
                - Single DataLoader if TTA is disabled
                - List of DataLoaders (one per TTA transform) if TTA is enabled
        """
        if self.test_ds is None:
            self.setup(stage='test')
            
        test_transforms = getattr(self.config, 'get_test_transforms', lambda: None)()
        
        # Handle TTA case
        if isinstance(test_transforms, list) and len(test_transforms) > 1:
            # TTA case - return list of DataLoaders with different transforms
            return self._create_tta_dataloaders(test_transforms)
        
        # Standard test case - single DataLoader
        kwargs = self._get_dataloader_kwargs()
        kwargs.update({
            'dataset': self.test_ds,
            'shuffle': False,
            'drop_last': False
        })
        
        return DataLoader(**kwargs)
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling class imbalance.
        
        Uses a smoothed version of inverse class frequency to handle imbalance.
        Weights are normalized to sum to the number of classes.
        
        Returns:
            torch.Tensor: Tensor of class weights
        """
        class_counts = self.train_df['label'].value_counts().sort_index()
        total_samples = len(self.train_df)
        num_classes = len(class_counts)
        
        # Calculate weights using inverse frequency with smoothing
        weights = total_samples / (num_classes * class_counts)
        
        # Normalize weights to sum to num_classes
        weights = weights / weights.sum() * num_classes
        
        # Convert to tensor
        self.class_weights = torch.tensor(weights.values, dtype=torch.float32)
        
        # Log class distribution
        class_dist = {}
        for cls, count in class_counts.items():
            weight = self.class_weights[self.class_to_idx[cls]].item()
            class_dist[cls] = {
                'count': count,
                'weight': f"{weight:.2f}",
                'percentage': f"{(count / total_samples * 100):.1f}%"
            }
        
        logging.info(f"Class distribution and weights: {json.dumps(class_dist, indent=2)}")
        
    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return len(self.class_to_idx) if self.class_to_idx else 0
        
    def get_class_names(self) -> List[str]:
        """Get the list of class names.
        
        Returns:
            List of class names in the order of their indices
        """
        return [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        
    def get_class_weights(self) -> torch.Tensor:
        """Get the class weights for handling class imbalance.
        
        Returns:
            Tensor of class weights
        """
        if self.class_weights is None:
            raise RuntimeError("Class weights not calculated. Call setup() first.")
        return self.class_weights
        
    def get_sample_counts(self) -> Dict[str, Dict[str, int]]:
        """Get the number of samples in each split.
        
        Returns:
            Dictionary with counts for 'train', 'val', and 'test' splits
        """
        return {
            'train': len(self.train_df) if self.train_df is not None else 0,
            'val': len(self.val_df) if self.val_df is not None else 0,
            'test': len(self.test_df) if self.test_df is not None else 0
        }
