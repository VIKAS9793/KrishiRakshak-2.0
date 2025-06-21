"""
Production-Ready Training Orchestrator for Plant Disease Classification.

This script includes an integrated callback for advanced visual logging,
removing the need for a separate callback file.
"""
"""
Train a deep learning model for plant disease classification.

This script handles the training pipeline including data loading, model training,
validation, and testing. It supports both RGB and multispectral data, and includes
features like mixed precision training, gradient accumulation, and learning rate scheduling.
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
import wandb
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from albumentations.pytorch import ToTensorV2
from matplotlib import cm
from PIL import Image
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

# Local application imports
from src.data.dataset import PlantDiseaseDataset
from src.models.model import PlantDiseaseModel
from src.utils.callbacks import (
    EarlyStopping,
    LearningRateLogger,
    ModelCheckpoint,
)
from src.utils.config import load_config, save_config
from src.utils.logger import setup_logging, logger
from src.utils.metrics import AverageMeter, accuracy, get_metrics
from src.utils.seed import set_seed

# Suppress some annoying warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.optim.lr_scheduler')
warnings.filterwarnings("ignore", category=UserWarning, module='pytorch_lightning')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

# Set matplotlib backend to 'Agg' to avoid GUI issues in headless environments
import matplotlib
matplotlib.use('Agg')

# To make this script runnable, install the project package
# using `pip install -e .` from the project root.
from src.data.dataset import PlantDiseaseDataset
from src.models.hybrid import HybridModel

logger = logging.getLogger(__name__)


# =============================================================================
# Integrated Callback for Advanced Visualization
# =============================================================================
class ImagePredictionLogger(Callback):
    """
    A callback defined directly in the training script to log model predictions
    and a confusion matrix during validation. It is logger-agnostic.

    Note: This callback requires the model's `validation_step` to return a dictionary
    containing the keys 'preds' and 'targets'.
    """
    def __init__(self, num_samples: int = 16):
        super().__init__()
        self.num_samples = num_samples
        self.validation_step_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Gather data from each validation step."""
        if not outputs:
            return

        # Store a limited number of samples to avoid high memory usage
        if len(self.validation_step_outputs) < self.num_samples:
            self.validation_step_outputs.append({
                "images": batch['image'].cpu(),
                "preds": outputs['preds'].cpu(),
                "targets": outputs['targets'].cpu()
            })

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log images and confusion matrix at the end of the validation epoch."""
        if not self.validation_step_outputs or not trainer.logger:
            self.validation_step_outputs.clear()
            return

        # --- Aggregate data from all collected batches ---
        images = torch.cat([x['images'] for x in self.validation_step_outputs])[:self.num_samples]
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).argmax(1)
        targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        # --- Log Prediction Visualizations ---
        # Reverse normalization for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = images * std + mean
        
        grid = make_grid(images, nrow=4)
        
        # Log to TensorBoard and WandB
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(
                    "Validation/Predictions", 
                    grid, 
                    global_step=trainer.current_epoch
                )
            elif isinstance(logger, WandbLogger):
                # Log images with predictions and ground truth
                captions = [
                    f"Pred: {class_names[pred]}\nTarget: {class_names[target]}"
                    for pred, target in zip(preds, targets)
                ]
                logger.log_image(
                    key="Validation/Predictions",
                    images=[grid],
                    caption=[f"Epoch {trainer.current_epoch}"],
                    step=trainer.current_epoch
                )
                
                # Log individual predictions
                logger.log({
                    "Validation/Prediction_Grid": wandb.Image(
                        grid,
                        caption=f"Epoch {trainer.current_epoch}"
                    )
                })

        # --- Log Confusion Matrix ---
        class_names = pl_module.hparams.get('class_names', [str(i) for i in range(pl_module.hparams.num_classes)])
        
        # Create confusion matrix plot
        cm = pd.crosstab(pd.Series(targets.numpy(), name='Actual'), pd.Series(preds.numpy(), name='Predicted'))
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Log the confusion matrix to all loggers
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure(
                    "Validation/Confusion_Matrix", 
                    fig, 
                    global_step=trainer.current_epoch
                )
            elif isinstance(logger, WandbLogger):
                logger.log({
                    "Validation/Confusion_Matrix": wandb.Image(fig),
                    "epoch": trainer.current_epoch
                })

        plt.close(fig) # Important to free memory
        self.validation_step_outputs.clear() # Clear memory for next epoch


# =============================================================================
# Main Script Functions
# =============================================================================
def get_data_loaders(config: dict):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config (dict): Configuration dictionary containing data and model settings.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders.
    """
    data_cfg = config['data']
    aug_cfg = config.get('data_augmentation', {})
    model_cfg = config['model']
    
    # Define image transformations
    image_size = data_cfg.get('image_size', [224, 224])
    
    # Training augmentations
    train_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.OneOf([
            A.RandomResizedCrop(height=image_size[0], width=image_size[1]),
            A.RandomCrop(height=image_size[0], width=image_size[1])
        ], p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1)
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5)
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5)
        ], p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    # Validation/Test transformations
    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    # Get dataset parameters
    use_ms = model_cfg.get('use_ms', False)
    ms_dir = data_cfg.get('ms_dir')
    ms_source = data_cfg.get('ms_source', 'synthetic')
    
    # Get metadata path
    metadata_path = Path(data_cfg['processed_dir']) / 'metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Run prepare_dataset.py first.")
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(
        csv_path=str(metadata_path),
        data_dir=Path(data_cfg['datasets']['plantvillage']['path']).parent,  # Parent of dataset dirs
        split='train',
        transform=train_transform,
        use_ms=use_ms,
        ms_source=ms_source,
        ms_dir=ms_dir
    )
    
    val_dataset = PlantDiseaseDataset(
        csv_path=str(metadata_path),
        data_dir=Path(data_cfg['datasets']['plantvillage']['path']).parent,  # Parent of dataset dirs
        split='val',
        transform=val_transform,
        use_ms=use_ms,
        ms_source=ms_source,
        ms_dir=ms_dir
    )
    
    test_dataset = PlantDiseaseDataset(
        csv_path=str(metadata_path),
        data_dir=Path(data_cfg['datasets']['plantvillage']['path']).parent,  # Parent of dataset dirs
        split='test',
        transform=val_transform,
        use_ms=use_ms,
        ms_source=ms_source,
        ms_dir=ms_dir
    )
    
    # Get data loader parameters
    batch_size = config['training'].get('batch_size', 32)
    num_workers = data_cfg.get('num_workers', 4)
    pin_memory = data_cfg.get('pin_memory', True)
    persistent_workers = data_cfg.get('persistent_workers', True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True  # Helps with batch norm stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    # Log dataset statistics
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    
    return train_loader, val_loader, test_loader

def train(config: dict):
    """Train the model with Weights & Biases integration."""
    proj_cfg = config['project']
    train_cfg = config['training']
    model_cfg = config['model']

    # Initialize wandb
    wandb_logger = WandbLogger(
        project="KrishiSahayak",
        name=proj_cfg['experiment_name'],
        log_model='all',
        config=config
    )

    try:
        log_dir = Path(proj_cfg['output_dir']) / proj_cfg['experiment_name']
        log_dir.mkdir(parents=True, exist_ok=True)
        
        pl.seed_everything(proj_cfg['seed'], workers=True)

        # Get data loaders
        train_loader, val_loader, test_loader = get_data_loaders(config)
        logger.info(f"DataLoaders created. Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset) if val_loader else 0}")

        # Log dataset statistics
        if hasattr(train_loader.dataset, 'class_to_idx'):
            class_names = list(train_loader.dataset.class_to_idx.keys())
            config['data']['class_names'] = class_names
            wandb_logger.experiment.config.update({"class_names": class_names}, allow_val_change=True)

        # Initialize model
        model_hparams = {
            'num_classes': model_cfg['num_classes'],
            'learning_rate': train_cfg['optimizer']['learning_rate'],
            'use_ms': model_cfg.get('use_ms', False),
            'backbone_name': model_cfg['backbone']['name'],
            'class_names': config['data'].get('class_names', [])
        }
        model = HybridModel(**model_hparams)
        logger.info(f"Model '{model.__class__.__name__}' with backbone '{model_cfg['backbone']['name']}' initialized.")

        # Log model architecture
        wandb_logger.watch(model, log='all', log_freq=100)

        # --- Callbacks ---
        checkpoint_dir = log_dir / 'checkpoints'
        checkpoint_cb = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=train_cfg['checkpoint']['monitor'],
            mode=train_cfg['checkpoint']['mode'],
            save_top_k=train_cfg['checkpoint'].get('save_top_k', 2),
            filename='{epoch}-{val_loss:.2f}-{val_accuracy:.2f}'
        )
        
        early_stop_cb = EarlyStopping(**train_cfg['early_stopping'])
        callbacks = [
            checkpoint_cb, 
            early_stop_cb, 
            LearningRateMonitor(logging_interval='step')
        ]
        
        # Add the integrated logger callback
        if config['logging'].get('log_visualizations', True):
            callbacks.append(ImagePredictionLogger())
        
        # Initialize trainer with wandb logger
        loggers = [
            wandb_logger,
            TensorBoardLogger(
                save_dir=log_dir, 
                name='', 
                version='',
                default_hp_metric=False
            ),
            CSVLogger(save_dir=log_dir, name='', version='')
        ]
        
        trainer = pl.Trainer(
            logger=loggers,
            callbacks=callbacks,
            max_epochs=train_cfg['max_epochs'],
            accelerator=config['hardware'].get('accelerator', 'auto'),
            devices=config['hardware'].get('devices', 'auto'),
            precision=str(train_cfg.get('precision', 32)),
            log_every_n_steps=train_cfg.get('log_interval', 50),
            gradient_clip_val=train_cfg.get('gradient_clip_val', 1.0),
            enable_progress_bar=True,
            enable_model_summary=True
        )

        # Log hyperparameters
        wandb_logger.log_hyperparams(config)

        logger.info("Starting model training...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        logger.info("Training finished.")

        # Test the model
        if test_loader is not None:
            logger.info("Starting final model testing...")
            test_metrics = trainer.test(dataloaders=test_loader, ckpt_path='best')
            wandb_logger.log_metrics({"test_metrics": test_metrics[0]})
            
            # Log final model
            model_path = checkpoint_dir / 'final_model.ckpt'
            trainer.save_checkpoint(model_path)
            wandb_logger.experiment.save(str(model_path))
            
        return model
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        if 'wandb_logger' in locals():
            wandb_logger.experiment.alert(
                title="Training Failed",
                text=str(e),
                level=wandb.AlertLevel.ERROR
            )
        raise
    finally:
        # Ensure wandb run is finished
        if 'wandb_logger' in locals():
            wandb.finish()

def main():
    """
    Main function to run the training process.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a plant disease classification model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode (fewer epochs, smaller batches, etc.)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply debug settings if needed
    if args.debug:
        logger.warning("Running in debug mode with reduced settings")
        config['training']['max_epochs'] = 2
        config['training']['batch_size'] = 8
        config['data']['num_workers'] = 2
        config['training']['fast_dev_run'] = False
        config['training']['limit_train_batches'] = 0.1
        config['training']['limit_val_batches'] = 0.1
        config['training']['limit_test_batches'] = 0.1
    
    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Set up logging
    setup_logging(config)
    
    # Log configuration
    logger.info(f"Starting training with configuration:\n{json.dumps(config, indent=2, default=str)}")
    
    # Log PyTorch and CUDA information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}")
    
    # Check if we're resuming from a checkpoint
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        
    try:
        # Train the model
        train(config)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.exception(f"Training failed with error: {str(e)}")
        raise

if __name__ == '__main__':
    main()