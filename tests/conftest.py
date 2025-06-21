"""
Pytest configuration and fixtures for KrishiSahayak tests.

This module contains fixtures and configuration that are shared across all tests.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch

from src import set_seed


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def tmp_dir() -> Generator[Path, None, None]:
    """Create and return a temporary directory for tests."""
    tmp_dir = tempfile.mkdtemp(prefix="krishisahayak_test_")
    yield Path(tmp_dir)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return the device to run tests on (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility in tests."""
    set_seed(42)


@pytest.fixture(scope="module")
def sample_image() -> torch.Tensor:
    """Return a sample image tensor for testing."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture(scope="module")
def sample_batch() -> dict:
    """Return a sample batch for testing."""
    return {
        "image": torch.randn(8, 3, 224, 224),  # Batch of 8 RGB images
        "label": torch.randint(0, 10, (8,)),  # 10 classes
    }
