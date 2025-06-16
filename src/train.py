"""Training script for KrishiRakshak plant disease classification."""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.config import Config
from src.data.datamodule import PlantDiseaseDataModule
from src.models.plant_model import PlantDiseaseModel


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train(config: Config) -> None:
    """Train the model."""
    # Set random seed
    set_seed(config.SEED)
    
    # Log hardware info
    config.log_hardware_info()
    
    # Data transformations
    train_transform = config.train_transform
    val_transform = config.val_transform
    test_transform = config.test_transform
    
    # Initialize data module
    datamodule = PlantDiseaseDataModule(
        config=config,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform
    )
    datamodule.setup()
    


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train KrishiRakshak model")
    parser.add_argument(
        "--model",
        type=str,
        default=Config.MODEL_NAME,
        help="Model architecture to use",
    )
    parser.add_argument(
        "--batch-size", type=int, default=Config.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=Config.EPOCHS, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=Config.LEARNING_RATE, help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = Config()
    config.MODEL_NAME = args.model
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Train the model
    train(config)
