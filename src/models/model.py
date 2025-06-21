"""
Plant Disease Classification Model with PyTorch Lightning.

This module defines the main model architecture for plant disease classification,
integrating with the training pipeline and supporting various backbones and configurations.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from torch import Tensor
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    _LRScheduler,
)
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

from .base import BaseModel
from ..utils.metrics import get_metrics

# Suppress some annoying warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.optim.lr_scheduler')
warnings.filterwarnings("ignore", category=UserWarning, module='pytorch_lightning')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class PlantDiseaseModel(BaseModel):
    """
    A flexible and production-ready model for plant disease classification.
    
    This model supports:
    - Multiple backbone architectures (via timm)
    - Mixed precision training
    - Multiple optimizers and learning rate schedulers
    - Class weights for imbalanced datasets
    - Gradient clipping
    - Advanced metrics tracking
    """
    
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-4,
        backbone_name: str = 'efficientnet_b0',
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        optimizer: str = 'AdamW',
        use_scheduler: bool = True,
        scheduler_type: str = 'cosine',
        class_weights: Optional[torch.Tensor] = None,
        weight_decay: float = 1e-4,
        gradient_clip_val: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the PlantDiseaseModel.
        
        Args:
            num_classes: Number of output classes
            learning_rate: Initial learning rate
            backbone_name: Name of the backbone architecture (from timm)
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout probability
            optimizer: Name of the optimizer to use ('Adam', 'AdamW', 'SGD')
            use_scheduler: Whether to use a learning rate scheduler
            scheduler_type: Type of scheduler to use ('cosine', 'step', 'plateau', 'one_cycle')
            class_weights: Optional tensor of class weights for imbalanced datasets
            weight_decay: Weight decay for the optimizer
            gradient_clip_val: If not None, clips gradient norm to this value
            **kwargs: Additional keyword arguments passed to the parent class
        """
        # Save hyperparameters to checkpoints
        self.save_hyperparameters()
        
        # Initialize parent class
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            optimizer=optimizer,
            use_scheduler=use_scheduler,
            scheduler_type=scheduler_type,
            class_weights=class_weights,
            weight_decay=weight_decay,
            gradient_clip_val=gradient_clip_val,
            dropout_rate=dropout_rate,
            **kwargs
        )
        
        # Load backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # We'll add our own head
            drop_rate=dropout_rate,
            **kwargs.get('backbone_kwargs', {})
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Create classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Model output logits of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Apply classification head
        logits = self.head(features)
        
        return logits
    
    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[Dict]]]:
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Single optimizer, or a list of optimizers with their schedulers.
        """
        # Get parameters to optimize (filter out parameters that don't require gradients)
        params = filter(lambda p: p.requires_grad, self.parameters())
        
        # Initialize optimizer
        if self.hparams.optimizer.lower() == 'adam':
            optimizer = Adam(
                params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == 'adamw':
            optimizer = AdamW(
                params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.lower() == 'sgd':
            optimizer = SGD(
                params,
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")
        
        # Configure learning rate scheduler
        if self.hparams.use_scheduler:
            if self.hparams.scheduler_type == 'cosine':
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.max_epochs,
                    eta_min=self.hparams.learning_rate * 1e-2
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'epoch',
                        'frequency': 1
                    }
                }
            elif self.hparams.scheduler_type == 'one_cycle':
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=self.hparams.learning_rate,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=len(self.train_dataloader())
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                        'frequency': 1
                    }
                }
            elif self.hparams.scheduler_type == 'plateau':
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.1,
                    patience=5,
                    verbose=True
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val_loss',
                        'interval': 'epoch',
                        'frequency': 1
                    }
                }
            else:
                raise ValueError(f"Unsupported scheduler type: {self.hparams.scheduler_type}")
        
        return optimizer
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Training step.
        
        Args:
            batch: Dictionary containing 'image' and 'label' keys
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with loss and metrics
        """
        # Forward pass
        logits = self(batch['image'])
        
        # Calculate loss
        loss = self.criterion(logits, batch['label'])
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(
            self.train_metrics(preds, batch['label']),
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        
        return {'loss': loss, 'preds': preds, 'targets': batch['label']}
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Validation step.
        
        Args:
            batch: Dictionary containing 'image' and 'label' keys
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with loss and metrics
        """
        # Forward pass
        logits = self(batch['image'])
        
        # Calculate loss
        loss = self.criterion(logits, batch['label'])
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            self.val_metrics(preds, batch['label']),
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        
        return {'loss': loss, 'preds': preds, 'targets': batch['label']}
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Test step.
        
        Args:
            batch: Dictionary containing 'image' and 'label' keys
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary with loss and metrics
        """
        # Forward pass
        logits = self(batch['image'])
        
        # Calculate loss
        loss = self.criterion(logits, batch['label'])
        
        # Get predictions and probabilities
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log_dict(
            self.test_metrics(preds, batch['label']),
            on_step=False,
            on_epoch=True
        )
        
        return {
            'loss': loss,
            'preds': preds,
            'probs': probs,
            'targets': batch['label']
        }
