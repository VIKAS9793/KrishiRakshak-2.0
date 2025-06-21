"""
Model implementations for plant disease classification.

This package contains the model architectures used in the KrishiSahayak project,
including the base model, hybrid RGB-multispectral model, and any future models.
"""

from .base import BaseModel
from .model import PlantDiseaseModel
from .hybrid import HybridModel

__all__ = [
    'BaseModel',
    'PlantDiseaseModel',
    'HybridModel',
]
