"""
Production-Grade Unit Test Suite for the KrishiSahayak HybridModel.

This test suite uses `pytest` and `unittest.mock` to ensure fast, reliable, and
isolated testing of the HybridModel's core logic and its interaction with the
data augmentation pipeline.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from unittest.mock import patch

# In a real setup, this is handled by installing the project package
# using `pip install -e .` from the project root.
from src.models.hybrid import HybridModel

# --- Mocking Fixtures and Dummy Models ---

class DummyBackbone(nn.Module):
    """A minimal dummy model to replace a real `timm` backbone during tests."""
    def __init__(self, num_features=1280):
        super().__init__()
        self.num_features = num_features
        self.conv_head = nn.Conv2d(3, self.num_features, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.avgpool(self.conv_head(x)).flatten(1)

@pytest.fixture
def mock_timm():
    """Pytest fixture to mock `timm.create_model` to prevent network calls."""
    with patch('src.models.hybrid.timm.create_model') as mock_create:
        mock_create.return_value = DummyBackbone()
        yield mock_create

@pytest.fixture
def dummy_batch():
    """Pytest fixture to provide a consistent dummy batch of data for model tests."""
    return {
        'image': torch.randn(4, 3, 224, 224),
        'ms_data': torch.randn(4, 4, 224, 224),
        'label': torch.randint(0, 38, (4,))
    }

# --- Test Cases ---

@pytest.mark.parametrize("use_ms", [True, False])
def test_model_instantiation(mock_timm, use_ms):
    """Tests if the HybridModel can be instantiated correctly in both modes."""
    model = HybridModel(num_classes=38, use_ms=use_ms, backbone_name='dummy_net')
    assert model is not None
    assert model.use_ms == use_ms
    if use_ms:
        assert hasattr(model, 'fusion')
    else:
        assert not hasattr(model, 'fusion')

@pytest.mark.parametrize("use_ms", [True, False])
def test_forward_pass_shape(mock_timm, dummy_batch, use_ms):
    """Tests the forward pass and validates the output shape."""
    batch_size = dummy_batch['image'].shape[0]
    num_classes = 38
    model = HybridModel(num_classes=num_classes, use_ms=use_ms)
    output = model(dummy_batch)
    assert output.shape == (batch_size, num_classes)

def test_gradient_flow(mock_timm, dummy_batch):
    """CRITICAL: Tests if gradients flow back to model parameters, ensuring it's trainable."""
    model = HybridModel(num_classes=38, use_ms=True)
    assert model.classifier.weight.grad is None

    output = model(dummy_batch)
    loss = nn.CrossEntropyLoss()(output, dummy_batch['label'])
    loss.backward()

    assert model.classifier.weight.grad is not None
    assert torch.all(model.classifier.weight.grad != 0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires a CUDA-enabled GPU")
def test_device_compatibility_cuda(mock_timm, dummy_batch):
    """Tests if the model runs correctly on a CUDA device."""
    device = torch.device("cuda")
    model = HybridModel(num_classes=38, use_ms=True).to(device)
    batch_on_device = {k: v.to(device) for k, v in dummy_batch.items()}
    
    output = model(batch_on_device)
    assert output.shape == (dummy_batch['image'].shape[0], 38)
    assert output.device.type == 'cuda'

def test_eval_mode_is_deterministic(mock_timm, dummy_batch):
    """Tests if the model output is the same across multiple calls in eval mode."""
    model = HybridModel(num_classes=38, use_ms=True)
    model.eval()

    with torch.no_grad():
        output1 = model(dummy_batch)
        output2 = model(dummy_batch)

    assert torch.allclose(output1, output2, atol=1e-6)

# --- Integrated Data Augmentation Test ---

def test_augmentation_synchronization_for_hybrid_data():
    """
    CRITICAL TEST: Ensures that random spatial augmentations are applied
    identically to both the RGB and MS image streams.
    """
    # Define a transform with a deterministic random spatial augmentation
    transform = A.Compose([
        A.HorizontalFlip(p=1.0),  # Always apply for a predictable test
        A.Rotate(limit=45, p=1.0)
    ], additional_targets={'ms_data': 'image'})

    # Create a non-symmetrical dummy image to clearly detect transformations
    image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    image_rgb[10:30, 10:90] = 255 # Asymmetrical rectangle
    
    image_ms = image_rgb.copy() # Start with an identical MS image

    # Apply the transform to the dictionary of images
    transformed_data = transform(image=image_rgb, ms_data=image_ms)
    
    transformed_rgb = transformed_data['image']
    transformed_ms = transformed_data['ms_data']

    # Assert that both images were transformed in the exact same way
    assert np.array_equal(transformed_rgb, transformed_ms), \
        "RGB and MS images received different augmentations."