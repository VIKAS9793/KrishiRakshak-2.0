"""Training script for KrishiRakshak plant disease classification."""
import argparse
import logging
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

# Assuming these are correctly defined in your src folder
from src.config import Config
from src.data.datamodule import PlantDiseaseDataModule
from src.models.plant_model import PlantDiseaseModel


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The following two lines are often recommended for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set to {seed}")


def train(config: Config) -> None:
    """
    Initializes and runs the model training pipeline.
    """
    # Set random seed for reproducibility
    set_seed(config.SEED)

    # Log hardware info
    config.log_hardware_info()

    # Initialize data module
    datamodule = PlantDiseaseDataModule(
        config=config,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_transform=config.train_transform,
        val_transform=config.val_transform,
        test_transform=config.test_transform,
    )
    # Setting up the datamodule is essential to get num_classes, etc.
    datamodule.setup()

    # Initialize model
    model = PlantDiseaseModel(
        model_name=config.MODEL_NAME,
        num_classes=datamodule.num_classes, # Get num_classes from data
        learning_rate=config.LEARNING_RATE,
        # You might need to pass more optimizer/scheduler configs here
    )

    # === Callbacks and Logger Setup ===
    log_dir = Path("logs/")
    log_dir.mkdir(exist_ok=True)
    
    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=log_dir, name=config.MODEL_NAME)
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{config.MODEL_NAME}/",
        filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    
    # Early stopping
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )

    # === Trainer Setup ===
    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        accelerator="auto", # Automatically chooses GPU/CPU/TPU
        devices="auto",
        logger=tensorboard_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar(),
        ],
        precision=config.PRECISION,
    )

    # --- Start Training ---
    logging.info(f"Starting training for model: {config.MODEL_NAME}")
    trainer.fit(model, datamodule=datamodule)

    # --- Optional: Run Testing ---
    logging.info("Training finished. Starting testing.")
    trainer.test(model, datamodule=datamodule, ckpt_path="best") # Use the best checkpoint


if __name__ == "__main__":
    # --- Setup Logging ---
    # Moved to the top to log everything
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler() # Also print to console
        ],
    )

    # --- Parse Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Train KrishiRakshak model.")
    # Defaults are still pulled from Config, which is fine
    parser.add_argument("--model", type=str, default=Config.MODEL_NAME, help="Model architecture.")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, help="Learning rate for the optimizer.")
    
    args = parser.parse_args()
    
    # --- Update Configuration ---
    # Create a single config object to pass around
    config = Config()
    config.MODEL_NAME = args.model
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    
    logging.info(f"Configuration loaded: {config.__dict__}")

    # --- Run Training ---
    train(config)
