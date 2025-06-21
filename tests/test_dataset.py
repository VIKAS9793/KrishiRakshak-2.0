"""Tests for the PlantDiseaseDataset class."""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import PlantDiseaseDataset


@pytest.fixture
def sample_metadata(tmp_path: Path) -> Dict[str, List[str]]:
    """Create a sample metadata DataFrame for testing."""
    # Create a temporary directory structure
    data_dir = tmp_path / "data"
    (data_dir / "train" / "healthy").mkdir(parents=True, exist_ok=True)
    (data_dir / "train" / "diseased").mkdir(parents=True, exist_ok=True)
    (data_dir / "val" / "healthy").mkdir(parents=True, exist_ok=True)
    (data_dir / "val" / "diseased").mkdir(parents=True, exist_ok=True)
    
    # Create some dummy image files
    image_paths = []
    labels = []
    splits = []
    
    # Create 4 train images (2 healthy, 2 diseased)
    for i in range(2):
        img_path = data_dir / "train" / "healthy" / f"healthy_{i}.jpg"
        img_path.touch()
        image_paths.append(str(img_path))
        labels.append("healthy")
        splits.append("train")
    
    for i in range(2):
        img_path = data_dir / "train" / "diseased" / f"diseased_{i}.jpg"
        img_path.touch()
        image_paths.append(str(img_path))
        labels.append("diseased")
        splits.append("train")
    
    # Create 2 validation images (1 healthy, 1 diseased)
    img_path = data_dir / "val" / "healthy" / "healthy_val.jpg"
    img_path.touch()
    image_paths.append(str(img_path))
    labels.append("healthy")
    splits.append("val")
    
    img_path = data_dir / "val" / "diseased" / "diseased_val.jpg"
    img_path.touch()
    image_paths.append(str(img_path))
    labels.append("diseased")
    splits.append("val")
    
    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'image_path': image_paths,
        'label': labels,
        'split': splits,
    })
    
    return {
        'metadata': metadata,
        'data_dir': str(data_dir),
    }


def test_plant_disease_dataset_init(sample_metadata):
    """Test that the dataset initializes correctly."""
    # Create dataset for training
    train_dataset = PlantDiseaseDataset(
        metadata=sample_metadata['metadata'],
        split='train',
        transform=None,
    )
    
    # Check that the dataset has the correct length
    assert len(train_dataset) == 4  # 2 healthy + 2 diseased
    
    # Check that the dataset has the expected attributes
    assert hasattr(train_dataset, 'image_paths')
    assert hasattr(train_dataset, 'labels')
    assert hasattr(train_dataset, 'class_to_idx')
    assert hasattr(train_dataset, 'transform')
    
    # Check class_to_idx mapping
    assert 'healthy' in train_dataset.class_to_idx
    assert 'diseased' in train_dataset.class_to_idx
    assert len(train_dataset.class_to_idx) == 2
    
    # Check validation dataset
    val_dataset = PlantDiseaseDataset(
        metadata=sample_metadata['metadata'],
        split='val',
        transform=None,
    )
    assert len(val_dataset) == 2  # 1 healthy + 1 diseased


def test_plant_disease_dataset_getitem(sample_metadata):
    """Test that __getitem__ returns the expected data."""
    # Create dataset with a simple transform
    dataset = PlantDiseaseDataset(
        metadata=sample_metadata['metadata'],
        split='train',
        transform=None,
    )
    
    # Get a sample
    sample = dataset[0]
    
    # Check that the sample has the expected keys
    assert 'image' in sample
    assert 'label' in sample
    assert 'image_path' in sample
    
    # Check that the label is an integer
    assert isinstance(sample['label'], int)
    
    # Check that the label is in the expected range
    assert 0 <= sample['label'] < len(dataset.class_to_idx)


def test_plant_disease_dataset_with_transforms(sample_metadata):
    """Test that transforms are applied correctly."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Define a simple transform
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Create dataset with transform
    dataset = PlantDiseaseDataset(
        metadata=sample_metadata['metadata'],
        split='train',
        transform=transform,
    )
    
    # Get a sample
    sample = dataset[0]
    
    # Check that the image has the expected shape and type
    image = sample['image']
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    assert image.shape == (3, 224, 224)  # C, H, W
    
    # Check that the pixel values are normalized
    assert image.min() >= -3.0  # Allow for some numerical error
    assert image.max() <= 3.0


def test_plant_disease_dataloader(sample_metadata):
    """Test that the dataset works with a DataLoader."""
    # Create dataset
    dataset = PlantDiseaseDataset(
        metadata=sample_metadata['metadata'],
        split='train',
        transform=None,
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Use 0 for tests to avoid issues with Windows
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Check batch structure
    assert 'image' in batch
    assert 'label' in batch
    assert 'image_path' in batch
    
    # Check batch shapes
    assert isinstance(batch['image'], torch.Tensor)
    assert batch['image'].shape == (2, 3, 256, 256)  # Default size is 256x256
    assert batch['label'].shape == (2,)
    assert len(batch['image_path']) == 2
