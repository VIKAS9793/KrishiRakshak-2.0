"""
Custom callbacks for PyTorch Lightning training.

This module contains various callbacks that can be used with PyTorch Lightning's
Trainer to enhance the training process with additional functionality.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    RichProgressBar,
    TQDMProgressBar,
)
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer

# Set up logger
logger = logging.getLogger(__name__)


class LearningRateLogger(Callback):
    """
    Logs the learning rate of each parameter group to the logger.
    
    This callback logs the learning rate(s) being used by the optimizer(s) during training.
    It's useful for monitoring how the learning rate changes over time, especially when
    using learning rate scheduling.
    """
    
    def __init__(self, logging_interval: str = 'epoch'):
        """
        Initialize the LearningRateLogger callback.
        
        Args:
            logging_interval: When to log the learning rate. Can be 'epoch' or 'step'.
        """
        super().__init__()
        self.logging_interval = logging_interval
        
        if self.logging_interval not in ['step', 'epoch']:
            raise ValueError("logging_interval should be 'step' or 'epoch'")
    
    def on_train_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: STEP_OUTPUT, 
        batch: Any, 
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log learning rate at the end of each training batch if interval is 'step'."""
        if self.logging_interval == 'step':
            self._log_lr(trainer, 'step')
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log learning rate at the end of each training epoch if interval is 'epoch'."""
        if self.logging_interval == 'epoch':
            self._log_lr(trainer, 'epoch')
    
    def _log_lr(self, trainer: Trainer, interval: str) -> None:
        """
        Log the learning rate for each parameter group.
        
        Args:
            trainer: The PyTorch Lightning trainer.
            interval: The current logging interval ('step' or 'epoch').
        """
        # Get the current optimizer
        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        
        if optimizer is None:
            return
        
        # Log learning rate for each parameter group
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group.get('lr')
            if lr is not None:
                trainer.logger.log_metrics(
                    {f'lr/group_{i}': lr},
                    step=trainer.global_step if interval == 'step' else trainer.current_epoch
                )


class GradientAccumulationScheduler(Callback):
    """
    Callback that schedules gradient accumulation over epochs or steps.
    
    This allows for changing the number of gradient accumulation steps during training,
    which can be useful for gradually increasing the effective batch size.
    """
    
    def __init__(
        self, 
        scheduling: Dict[int, int],
        update_interval: str = 'epoch',
    ) -> None:
        """
        Initialize the GradientAccumulationScheduler callback.
        
        Args:
            scheduling: Dictionary mapping epoch/step numbers to the number of gradient
                       accumulation steps to use. For example, {0: 1, 5: 2} would use
                       1 accumulation step for the first 5 epochs/steps, then switch to 2.
            update_interval: When to update the accumulation ('epoch' or 'step').
        """
        super().__init__()
        self.scheduling = scheduling
        self.update_interval = update_interval
        
        if self.update_interval not in ['step', 'epoch']:
            raise ValueError("update_interval should be 'step' or 'epoch'")
        
        # Sort the scheduling dictionary by key
        self.scheduling = dict(sorted(self.scheduling.items()))
    
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Update gradient accumulation at the start of each epoch if interval is 'epoch'."""
        if self.update_interval == 'epoch':
            self._update_accumulation(trainer, 'epoch')
    
    def on_train_batch_start(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Update gradient accumulation at the start of each batch if interval is 'step'."""
        if self.update_interval == 'step':
            self._update_accumulation(trainer, 'step')
    
    def _update_accumulation(self, trainer: Trainer, interval: str) -> None:
        """
        Update the number of gradient accumulation steps based on the current epoch/step.
        
        Args:
            trainer: The PyTorch Lightning trainer.
            interval: The current update interval ('epoch' or 'step').
        """
        current = trainer.current_epoch if interval == 'epoch' else trainer.global_step
        
        # Find the appropriate accumulation value
        accumulation = 1  # Default value
        for key in sorted(self.scheduling.keys()):
            if current >= key:
                accumulation = self.scheduling[key]
            else:
                break
        
        # Update the trainer's accumulation_scheduler if it exists
        if hasattr(trainer, 'accumulation_scheduler'):
            trainer.accumulation_scheduler.scheduling = {0: accumulation}
            trainer.accumulation_scheduler._accumulate_grad_batches = accumulation
        
        # Log the change
        logger.info(f"Updated gradient accumulation to {accumulation} at {interval} {current}")


class ModelCheckpointWithBestScore(ModelCheckpoint):
    """
    Extended ModelCheckpoint that tracks and logs the best score achieved during training.
    
    This callback extends the standard ModelCheckpoint to provide additional functionality
    like tracking the best score and saving additional metadata.
    """
    
    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        **kwargs,
    ):
        """
        Initialize the ModelCheckpointWithBestScore callback.
        
        Args are the same as the parent ModelCheckpoint class.
        """
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            **kwargs,
        )
        
        self.best_score = None
        self.best_epoch = -1
        self.best_global_step = -1
    
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the validation loop ends."""
        super().on_validation_end(trainer, pl_module)
        
        # Update best score information
        if self.monitor:
            current = trainer.callback_metrics.get(self.monitor)
            if current is not None:
                if self.best_score is None or (
                    (self.monitor_op(current, self.best_score) and self.best_score != current)
                ):
                    self.best_score = current
                    self.best_epoch = trainer.current_epoch
                    self.best_global_step = trainer.global_step
                    
                    # Log the new best score
                    trainer.logger.log_metrics(
                        {f'best_{self.monitor}': self.best_score},
                        step=trainer.global_step
                    )
                    
                    # Log additional information
                    trainer.logger.log_metrics(
                        {
                            f'best_{self.monitor}_epoch': self.best_epoch,
                            f'best_{self.monitor}_step': self.best_global_step,
                        },
                        step=trainer.global_step
                    )


def get_default_callbacks(
    monitor: str = 'val_loss',
    mode: str = 'min',
    save_top_k: int = 1,
    verbose: bool = True,
    patience: int = 10,
    dirpath: Optional[Union[str, Path]] = None,
    filename: Optional[str] = '{epoch}-{val_loss:.4f}-{val_accuracy:.4f}',
    save_weights_only: bool = False,
    save_last: bool = True,
    log_lr: bool = True,
    log_graph: bool = True,
) -> List[Callback]:
    """
    Get a list of default callbacks for training.
    
    Args:
        monitor: Metric to monitor for checkpointing and early stopping.
        mode: One of 'min' or 'max' (minimize or maximize the monitored metric).
        save_top_k: Save the top k models according to the monitored metric.
        verbose: Whether to print messages when saving/validating.
        patience: Number of epochs with no improvement after which training will be stopped.
        dirpath: Directory to save checkpoints. If None, uses 'checkpoints' in the current directory.
        filename: Checkpoint filename. Can include placeholders like {epoch}, {val_loss}, etc.
        save_weights_only: If True, only the model's weights will be saved.
        save_last: If True, always save a checkpoint at the end of every epoch.
        log_lr: If True, include the LearningRateLogger callback.
        log_graph: If True, log the model graph at the start of training.
        
    Returns:
        List of callbacks to be used with the Trainer.
    """
    callbacks = []
    
    # Set up checkpoint directory
    if dirpath is None:
        dirpath = Path('checkpoints')
    else:
        dirpath = Path(dirpath)
    
    dirpath.mkdir(parents=True, exist_ok=True)
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpointWithBestScore(
        dirpath=dirpath,
        filename=filename,
        monitor=monitor,
        verbose=verbose,
        save_top_k=save_top_k,
        save_weights_only=save_weights_only,
        mode=mode,
        save_last=save_last,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=monitor,
        min_delta=0.00,
        patience=patience,
        verbose=verbose,
        mode=mode,
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    if log_lr:
        lr_monitor = LearningRateLogger(logging_interval='step')
        callbacks.append(lr_monitor)
    
    # Progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    return callbacks
