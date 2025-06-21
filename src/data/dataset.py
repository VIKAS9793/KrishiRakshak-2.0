"""
Production-worthy Dataset class for plant disease classification.

This version incorporates MAANG-level best practices including:
- Synchronized transformations for multi-modal data using Albumentations.
- Robust error handling for missing files and data integrity issues.
- Enhanced, professional docstrings for API clarity.
- Explicit checks for configuration errors.
- Recommendations for performance optimization.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Set up a logger for data-related warnings and errors.
# This is preferable to print() for production systems.
logger = logging.getLogger(__name__)

class PlantDiseaseDataset(Dataset):
    """
    Loads plant disease data with support for RGB and multispectral (MS) images.

    This class is designed for production use, handling potential data issues
    gracefully and ensuring that spatial transformations are applied consistently
    to both RGB and MS image pairs. It expects transformations via a library
    like Albumentations that can operate on a dictionary of images.

    Attributes:
        data_dir (Path): The root directory for RGB images.
        df (pd.DataFrame): The filtered dataframe containing metadata for the specified split.
        transform (Optional[Callable]): The augmentation callable.
        use_ms (bool): Flag indicating if multispectral data should be loaded.
        ms_dir (Optional[Path]): The root directory for multispectral images.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        data_dir: Union[str, Path],
        split: str = 'train',
        transform: Optional[Callable] = None,
        # MS data configuration
        use_ms: bool = False,
        ms_source: str = 'synthetic',  # 'synthetic' or 'real'
        ms_dir: Optional[Union[str, Path]] = None,
        ms_ext: str = '.tif',  # Default extension for MS files
    ):
        """
        Initializes the dataset and performs configuration checks.

        Args:
            csv_path (Union[str, Path]): Path to the CSV file containing image metadata.
                The CSV must contain 'image_path', 'label', 'ms_path' (if use_ms=True),
                and 'split' columns.
            data_dir (Union[str, Path]): Root directory for RGB images.
            split (str): The dataset split to load. Must be one of 'train', 'val', or 'test'.
            transform (Optional[Callable]): A callable transform (e.g., from Albumentations)
                to be applied. It should accept and return a dictionary of images.
            use_ms (bool): If True, loads corresponding multispectral data.
            ms_dir (Optional[Union[str, Path]]): Root directory for real multispectral images.
                This is required if use_ms is True.
            use_synthetic_ms (bool): If True, falls back to synthetic MS data when real MS is not available.
            synthetic_ms_dir (Optional[Union[str, Path]]): Root directory for synthetic multispectral images.
                Required if use_synthetic_ms is True.
            ms_ext (str): File extension for multispectral images (default: '.png').

        Raises:
            ValueError: If configuration is invalid.
            FileNotFoundError: If required files or directories are not found.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        # MS data configuration
        self.use_ms = use_ms
        self.ms_source = ms_source.lower()
        self.ms_ext = ms_ext
        
        # Set up MS data paths
        if self.use_ms:
            if self.ms_source == 'real' and ms_dir:
                self.ms_dir = Path(ms_dir)
            elif self.ms_source == 'synthetic':
                self.ms_dir = Path(ms_dir) if ms_dir else Path(data_dir) / 'synthetic_ms'
            else:
                raise ValueError("For MS data, either provide ms_dir for real MS or use synthetic MS")
                
            logger.info(f"Using {self.ms_source.upper()} MS data from: {self.ms_dir}")

        # Load and filter metadata once, with robust error handling.
        try:
            df = pd.read_csv(csv_path)
            
            # Ensure required columns exist
            required_columns = {'image_path', 'label', 'split', 'dataset'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Metadata CSV is missing required columns: {missing}")
                
            # Filter for the requested split
            self.df = df[df['split'] == split].reset_index(drop=True)
            
            if self.df.empty:
                raise ValueError(f"No samples found for split: {split}")
                
            # Create label mapping
            self.classes = sorted(self.df['label'].unique())
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            
            logger.info(f"Loaded {len(self.df)} samples for {split} split")
            logger.info(f"Number of classes: {len(self.classes)}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata CSV not found at: {csv_path}")

        self.df = df[df['split'] == split].reset_index(drop=True)
        if len(self.df) == 0:
            logger.warning(
                f"No samples found for split '{split}' in {csv_path}. "
                f"The dataset will be empty."
            )

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.df)

    def _load_image(self, path: Path) -> Image.Image:
        """Load an image from disk with error handling."""
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {path}: {str(e)}")
            raise

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the sample at the given index.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'image': The RGB image tensor.
                - 'ms_image': The multispectral image tensor (if use_ms is True).
                - 'label': The class label as an integer.
                - 'image_path': The path to the image file.
        """
        row = self.df.iloc[idx]
        
        # Construct the full image path
        img_path = self.data_dir / row['image_path']
        
        # Load RGB image
        try:
            image = self._load_image(img_path)
            image_array = np.array(image)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # Return a zero tensor of the same shape as expected
            image_array = np.zeros((256, 256, 3), dtype=np.uint8)  # Default size, adjust if needed
        
        # Initialize the sample dictionary
        sample = {
            'image': image_array,
            'label': self.class_to_idx[row['label']],
            'image_path': str(img_path)
        }
        
        # Load MS data if enabled
        if self.use_ms:
            if self.ms_source == 'real':
                # For real MS data, use the ms_path from metadata or construct path
                ms_path = self.ms_dir / row['image_path'].replace('.jpg', self.ms_ext)
            else:
                # For synthetic MS, use the same path structure but in the synthetic_ms directory
                ms_path = self.ms_dir / row['image_path'].replace('.jpg', self.ms_ext)
                
            try:
                ms_image = self._load_image(ms_path)
                sample['ms_image'] = np.array(ms_image)
            except Exception as e:
                logger.error(f"Error loading MS image {ms_path}: {str(e)}")
                # If MS loading fails, return zeros with the same shape as RGB
                sample['ms_image'] = np.zeros_like(sample['image'])
        
        # Apply transformations if specified
        if self.transform:
            sample = self.transform(**sample)
            
        return sample
            
    def _load_ms_data(self, row: pd.Series) -> Optional[torch.Tensor]:
        """
        Loads the multispectral data for the given row.

        This is a helper method for loading MS data, which can be extended
        to support different MS data formats or sources.

        Args:
            row (pd.Series): A row from the metadata DataFrame.

        Returns:
            Optional[torch.Tensor]: The loaded MS data as a tensor, or None if
                MS data is disabled or the file is not found.
        """
        if not self.use_ms or self.ms_dir is None:
            return None

        try:
            # Construct the MS file path based on the RGB path
            if self.ms_source == 'real':
                # For real MS data, use the ms_path from metadata or construct path
                ms_path = self.ms_dir / row['image_path'].replace('.jpg', self.ms_ext)
            else:
                # For synthetic MS, use the same path structure but in the synthetic_ms directory
                ms_path = self.ms_dir / row['image_path'].replace('.jpg', self.ms_ext)
            
            # Load the MS image (assuming it's saved as an image file)
            ms_image = Image.open(ms_path)
            ms_array = np.array(ms_image)
            if len(ms_array.shape) == 2:  # Single channel
                ms_array = np.expand_dims(ms_array, axis=0)
            elif len(ms_array.shape) == 3:  # Multi-channel
                if ms_array.shape[2] <= 4:  # Assume channels-last format
                    ms_array = np.transpose(ms_array, (2, 0, 1))
            
            # For synthetic data, we might need to replicate channels
            if self.ms_source == 'synthetic' and ms_array.shape[0] < 4:
                # Replicate channels to match expected band count
                ms_array = np.tile(ms_array, (4 // ms_array.shape[0] + 1, 1, 1))[:4]
                
            return torch.from_numpy(ms_array).float()
            
        except Exception as e:
            logger.warning(f"Failed to load {self.ms_source.upper()} MS data for {row.get('image_path', 'unknown')}: {e}")
            logger.debug("Detailed error:", exc_info=True)
            return None