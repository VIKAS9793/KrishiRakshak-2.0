"""
Modern Training Script for KrishiRakshak Plant Disease Classification
Following 2024-2025 best practices
"""
import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Local imports
from src.config import Config
from src.data.datamodule import PlantDiseaseDataModule
from src.models.plant_model import PlantDiseaseModel


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler("logs/training.log"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing configuration
    )
    
    # Suppress noisy logs
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # For full reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Set PyTorch Lightning seed
    pl.seed_everything(seed, workers=True)
    
    logging.info(f"Random seed set to {seed}")


def create_callbacks(config: Config) -> List[pl.Callback]:
    """Create and configure training callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        filename=f"{config.MODEL_NAME}-{{epoch:02d}}-{{val_loss:.2f}}-{{val_acc:.2f}}",
        monitor=config.MONITOR_METRIC,
        mode=config.MODE,
        save_top_k=config.CHECKPOINT_SAVE_TOP_K,
        save_last=True,
        save_on_train_epoch_end=False,  # Save on validation end
        verbose=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=config.MONITOR_METRIC,
        patience=config.EARLY_STOPPING_PATIENCE,
        mode=config.MODE,
        min_delta=config.MIN_DELTA,
        verbose=True,
        strict=True,  # Crash if monitor metric is not found
        check_finite=True  # Stop if monitor becomes NaN/inf
    )
    callbacks.append(early_stopping)
    
    # Model summary
    callbacks.append(ModelSummary(max_depth=3))
    
    # Rich progress bar for better visualization
    callbacks.append(RichProgressBar())
    
    # Stochastic Weight Averaging (if enabled)
    if getattr(config, 'USE_SWA', False):
        swa_callback = StochasticWeightAveraging(
            swa_lrs=config.LEARNING_RATE * 0.1,  # Typically lower than base LR
            swa_epoch_start=0.8,  # Start SWA at 80% of training
            annealing_epochs=10
        )
        callbacks.append(swa_callback)
        logging.info("Stochastic Weight Averaging enabled")
    
    return callbacks


def create_loggers(config: Config) -> List[pl.loggers.Logger]:
    """Create and configure loggers."""
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=str(config.LOGS_DIR),
        name=config.EXPERIMENT_NAME,
        version=f"{config.MODEL_NAME}-{os.environ.get('SLURM_JOB_ID', 'local')}",
        log_graph=True,
        default_hp_metric=False
    )
    loggers.append(tb_logger)
    
    # Weights & Biases logger (if enabled)
    if getattr(config, 'USE_WANDB', False):
        try:
            wandb_logger = WandbLogger(
                project=config.WANDB_PROJECT,
                entity=getattr(config, 'WANDB_ENTITY', None),
                name=f"{config.EXPERIMENT_NAME}-{config.MODEL_NAME}",
                log_model="all",
                save_dir=str(config.LOGS_DIR)
            )
            loggers.append(wandb_logger)
            logging.info("Weights & Biases logging enabled")
        except ImportError:
            logging.warning("wandb not installed, skipping W&B logging")
    
    return loggers


def train(config: Config, resume_from: Optional[str] = None) -> None:
    """
    Modern training pipeline with advanced features.
    """
    logging.info("Starting training pipeline...")
    config.log_hardware_info()
    
    # Initialize data module
    logging.info("Initializing data module...")
    datamodule = PlantDiseaseDataModule(
        config=config,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_transform=config.get_train_transforms(),
        val_transform=config.get_val_transforms(),
        test_transform=config.get_test_transforms(),
        persistent_workers=getattr(config, 'PERSISTENT_WORKERS', True),
        pin_memory=getattr(config, 'PIN_MEMORY', True)
    )
    
    # Setup data - this will initialize train/val/test datasets
    datamodule.setup()
    logging.info(f"Data loaded: {len(datamodule.train_ds)} train, "
                f"{len(datamodule.val_ds)} val, "
                f"{len(datamodule.test_ds)} test samples")
    
    # Initialize model
    logging.info(f"Initializing model: {config.MODEL_NAME}")
    model = PlantDiseaseModel(
        model_name=config.MODEL_NAME,
        num_classes=datamodule.num_classes,
        learning_rate=config.LEARNING_RATE,
        weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-4),
        dropout_rate=getattr(config, 'DROPOUT_RATE', 0.1),
        label_smoothing=getattr(config, 'LABEL_SMOOTHING', 0.0),
        pretrained=getattr(config, 'PRETRAINED', True)
    )
    
    # Create callbacks and loggers
    callbacks = create_callbacks(config)
    loggers = create_loggers(config)
    
    # Trainer setup
    trainer = pl.Trainer(
        # Training configuration
        max_epochs=getattr(config, 'NUM_EPOCHS', config.EPOCHS),
        min_epochs=getattr(config, 'MIN_EPOCHS', 1),
        accelerator="auto",
        devices="auto",
        strategy="auto",  # Automatically choose best strategy
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm="norm",
        precision=config.PRECISION,
        compile=config.COMPILE_MODEL,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        deterministic=config.DETERMINISTIC,
        benchmark=config.BENCHMARK,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        enable_model_summary=True,
        fast_dev_run=config.FAST_DEV_RUN if hasattr(config, 'FAST_DEV_RUN') else False,
        overfit_batches=config.OVERFIT_BATCHES if hasattr(config, 'OVERFIT_BATCHES') else 0.0,
        limit_train_batches=config.LIMIT_TRAIN_BATCHES if hasattr(config, 'LIMIT_TRAIN_BATCHES') else 1.0,
        limit_val_batches=config.LIMIT_VAL_BATCHES if hasattr(config, 'LIMIT_VAL_BATCHES') else 1.0,
        limit_test_batches=config.LIMIT_TEST_BATCHES if hasattr(config, 'LIMIT_TEST_BATCHES') else 1.0,
    )
    
    # Log model architecture
    try:
        if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'add_graph'):
            sample_batch = next(iter(datamodule.train_dataloader()))
            
            # Handle different batch formats
            if isinstance(sample_batch, dict):
                sample_input = sample_batch['image'][:1]
            elif isinstance(sample_batch, (list, tuple)):
                sample_input = sample_batch[0][:1]  # Assume image is first element
            else:
                sample_input = sample_batch[:1]  # Try direct indexing as last resort
                
            # Move to device if needed
            if hasattr(model, 'device'):
                sample_input = sample_input.to(model.device)
                
            trainer.logger.experiment.add_graph(model, sample_input)
            logging.info("Successfully logged model graph")
    except Exception as e:
        logging.warning(f"Could not log model graph: {e}", exc_info=True)
    
    # Training loop
    try:
        logging.info("Starting training...")
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=resume_from
        )
        
        # Test the model on best checkpoint
        if not trainer.fast_dev_run and trainer.checkpoint_callback.best_model_path:
            logging.info("Starting testing on best model...")
            trainer.test(
                ckpt_path="best",
                datamodule=datamodule
            )
        
        logging.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user")
        return
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        for logger in loggers:
            if hasattr(logger, 'save'):
                logger.save()
            if hasattr(logger, 'finalize'):
                status = "success" if trainer.checkpoint_callback.best_model_path else "failed"
                logger.finalize(status)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train KrishiRakshak plant disease classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model", type=str, help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, help="Weight decay")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    
    # System configuration
    parser.add_argument("--num-workers", type=int, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--precision", type=str, choices=["16", "32", "16-mixed", "bf16-mixed"], 
                       help="Training precision")
    
    # Resuming and debugging
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--fast-dev-run", action="store_true", help="Fast development run")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Setup logging
        setup_logging(log_level=getattr(args, 'log_level', 'INFO'))
        logging.info("Starting KrishiRakshak training")
        
        # Initialize config
        config = Config()
        
        # Update config from file if provided
        if args.config:
            logging.info(f"Loading configuration from {args.config}")
            config.update_from_file(args.config)
        
        # Override config with command line arguments
        for arg, value in vars(args).items():
            if value is not None and hasattr(config, arg.upper()):
                setattr(config, arg.upper(), value)
                logging.info(f"Config override: {arg.upper()} = {value}")
        
        # Set debug mode
        if args.debug:
            config.FAST_DEV_RUN = True
            config.NUM_EPOCHS = 2
            config.LOG_EVERY_N_STEPS = 1
            logging.info("Debug mode enabled")
        
        # Set random seed early for reproducibility
        seed = getattr(config, 'SEED', 42)
        set_seed(seed)
        
        # Create all necessary directories
        logging.info("Setting up project directories...")
        config.setup_directories()
        
        # Log final configuration
        logging.info("Final configuration:")
        for attr_name in sorted(dir(config)):
            if attr_name.isupper() and not attr_name.startswith('_'):
                logging.info(f"  {attr_name}: {getattr(config, attr_name)}")
        
        # Start training
        train(config, resume_from=args.resume)
        
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logging.info("Training script finished")


if __name__ == "__main__":
    main()
