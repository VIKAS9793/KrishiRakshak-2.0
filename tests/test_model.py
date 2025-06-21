"""Tests for the PlantDiseaseModel class."""

import pytest
import torch
from torch import nn

from src.models import PlantDiseaseModel


def test_plant_disease_model_initialization():
    """Test that the model initializes correctly."""
    # Create a model with a small number of classes for testing
    model = PlantDiseaseModel(num_classes=5)
    
    # Check that the model has the expected attributes
    assert hasattr(model, 'backbone')
    assert hasattr(model, 'head')
    assert isinstance(model.head, nn.Sequential)
    
    # Check that the model is in training mode by default
    assert model.training


def test_plant_disease_model_forward(sample_image):
    """Test the forward pass of the model."""
    model = PlantDiseaseModel(num_classes=10)
    
    # Move model to the same device as the input
    device = sample_image.device
    model = model.to(device)
    
    # Forward pass
    output = model(sample_image)
    
    # Check output shape
    assert output.shape == (1, 10)  # Batch size 1, 10 classes


def test_plant_disease_model_training_step(sample_batch):
    """Test a single training step."""
    model = PlantDiseaseModel(num_classes=10)
    
    # Move model and batch to the same device
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in sample_batch.items()}
    
    # Training step
    output = model.training_step(batch, batch_idx=0)
    
    # Check that the output contains the expected keys
    assert 'loss' in output
    assert 'preds' in output
    assert 'targets' in output
    
    # Check that the loss is a scalar tensor
    assert isinstance(output['loss'], torch.Tensor)
    assert output['loss'].dim() == 0  # Scalar


def test_plant_disease_model_validation_step(sample_batch):
    """Test a single validation step."""
    model = PlantDiseaseModel(num_classes=10)
    
    # Move model and batch to the same device
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in sample_batch.items()}
    
    # Validation step
    output = model.validation_step(batch, batch_idx=0)
    
    # Check that the output contains the expected keys
    assert 'val_loss' in output
    assert 'preds' in output
    assert 'targets' in output


def test_plant_disease_model_configure_optimizers():
    """Test that the model configures optimizers correctly."""
    # Test with default optimizer (AdamW)
    model = PlantDiseaseModel(num_classes=10)
    optimizer = model.configure_optimizers()
    
    # Should return a single optimizer by default
    if isinstance(optimizer, (list, tuple)):
        optimizer = optimizer[0]  # In case it's a list with length 1
    
    assert isinstance(optimizer, torch.optim.Optimizer)
    
    # Test with scheduler
    model = PlantDiseaseModel(
        num_classes=10,
        use_scheduler=True,
        scheduler_type='cosine',
    )
    optimizers = model.configure_optimizers()
    assert isinstance(optimizers, dict)
    assert 'optimizer' in optimizers
    assert 'lr_scheduler' in optimizers


@pytest.mark.parametrize('optimizer_name', ['Adam', 'AdamW', 'SGD'])
def test_different_optimizers(optimizer_name):
    """Test that different optimizers can be used."""
    model = PlantDiseaseModel(
        num_classes=10,
        optimizer=optimizer_name,
    )
    
    # This will raise an exception if the optimizer is not recognized
    optimizer = model.configure_optimizers()
    if isinstance(optimizer, (list, tuple)):
        optimizer = optimizer[0]  # In case it's a list with length 1
    
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.Optimizer)
