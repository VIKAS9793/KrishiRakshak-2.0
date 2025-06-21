"""
Abstract base model class for all plant disease classification models.

This version incorporates MAANG-level best practices including:
- Abstract Base Class (ABC) to enforce a clear implementation contract.
- Integration with `torchmetrics` for comprehensive performance tracking.
- Centralized step logic to avoid code duplication.
- Robust optimizer and learning rate scheduler configuration.
- Support for dictionary-based batches from the DataLoader.
- Enhanced error handling and validation.
- Support for class weights and additional optimizers.
- Gradient clipping and mixed precision training support.
"""
import abc
from typing import Any, Dict, Optional, Union
import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


class BaseModel(pl.LightningModule, metaclass=abc.ABCMeta):
    """
    Abstract Base Class for all models in the project.

    It enforces a contract for child classes to implement their specific
    architecture while providing shared, production-ready logic for training,
    validation, testing, and optimization.

    Attributes:
        criterion: The loss function.
        train_metrics: A collection of metrics for the training phase.
        val_metrics: A collection of metrics for the validation phase.
        test_metrics: A collection of metrics for the test phase.
    """

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-4,
        optimizer: str = "AdamW",
        use_scheduler: bool = True,
        scheduler_type: str = "cosine",
        class_weights: Optional[torch.Tensor] = None,
        weight_decay: float = 1e-4,
        gradient_clip_val: Optional[float] = None,
        dropout_rate: float = 0.1,
    ):
        """
        Initializes the BaseModel.

        Args:
            num_classes (int): The number of target classes.
            learning_rate (float): The learning rate for the optimizer.
            optimizer (str): The name of the optimizer to use ("AdamW", "Adam", "SGD").
            use_scheduler (bool): Flag to enable/disable the learning rate scheduler.
            scheduler_type (str): Type of scheduler ("cosine", "step", "plateau").
            class_weights (Optional[torch.Tensor]): Weights for imbalanced classes.
            weight_decay (float): Weight decay for regularization.
            gradient_clip_val (Optional[float]): Value for gradient clipping.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        
        # Validate inputs
        self._validate_inputs(num_classes, learning_rate, optimizer, scheduler_type)
        
        # Saves hyperparameters to the checkpoint, essential for reproducibility.
        self.save_hyperparameters()

        # Define the loss function with optional class weights for imbalanced datasets.
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # --- Metrics Configuration ---
        # Using MetricCollection is efficient and handles logging automatically.
        common_metrics = MetricCollection({
            'accuracy': Accuracy(task="multiclass", num_classes=num_classes),
            'precision': Precision(task="multiclass", num_classes=num_classes, average='macro'),
            'recall': Recall(task="multiclass", num_classes=num_classes, average='macro'),
            'f1_score': F1Score(task="multiclass", num_classes=num_classes, average='macro'),
        })
        self.train_metrics = common_metrics.clone(prefix='train_')
        self.val_metrics = common_metrics.clone(prefix='val_')
        self.test_metrics = common_metrics.clone(prefix='test_')

        # --- Model Architecture (to be defined in child classes) ---
        self.feature_extractor = None
        self.classifier = None
        self._build_model()  # Call the abstract method to build the architecture.

    def _validate_inputs(self, num_classes: int, learning_rate: float, 
                        optimizer: str, scheduler_type: str) -> None:
        """Validates input parameters."""
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if optimizer not in ["AdamW", "Adam", "SGD"]:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        if scheduler_type not in ["cosine", "step", "plateau"]:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    @abc.abstractmethod
    def _build_model(self):
        """
        Builds the model architecture.

        This abstract method must be implemented by all child classes to define
        `self.feature_extractor` and `self.classifier`.
        """
        raise NotImplementedError("Child classes must implement _build_model method")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output logits from the classifier.
        """
        if self.feature_extractor is None or self.classifier is None:
            raise RuntimeError(
                "Model architecture not defined. Ensure _build_model is properly implemented."
            )
        
        try:
            features = self.feature_extractor(x)
            # Adaptive pooling to handle variable feature map sizes
            if len(features.shape) > 2:
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            return self.classifier(features)
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {str(e)}")

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        """
        Performs a shared step for training, validation, and testing.

        This method handles dictionary-based batches, calculates loss,
        and updates the relevant metrics.

        Args:
            batch (Dict[str, Any]): The output from the DataLoader, expected to be a dictionary.
            stage (str): The current stage ('train', 'val', or 'test').

        Returns:
            torch.Tensor: The calculated loss for the batch.
        """
        # Validate batch structure
        if not isinstance(batch, dict):
            raise TypeError("Batch must be a dictionary")
        if 'image' not in batch or 'label' not in batch:
            raise KeyError("Batch must contain 'image' and 'label' keys")

        x = batch['image']
        y = batch['label']

        # Validate tensor shapes
        if x.dim() != 4:  # [batch_size, channels, height, width]
            raise ValueError(f"Expected 4D image tensor, got {x.dim()}D")
        if y.dim() != 1:  # [batch_size]
            raise ValueError(f"Expected 1D label tensor, got {y.dim()}D")

        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Select the correct metric collection based on the stage.
        metrics = getattr(self, f"{stage}_metrics")
        metrics.update(y_hat, y)

        # Log loss and metrics with appropriate sync settings for distributed training
        sync_dist = self.trainer.num_devices > 1 if self.trainer else False
        
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, 
                prog_bar=True, sync_dist=sync_dist)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=sync_dist)
        
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Performs a single validation step."""
        self._shared_step(batch, 'val')

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Performs a single test step."""
        self._shared_step(batch, 'test')

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs prediction step for inference."""
        x = batch['image']
        return torch.softmax(self(x), dim=1)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the optimizer and an optional learning rate scheduler.

        Returns:
            Dict[str, Any]: A dictionary containing the optimizer and lr_scheduler configuration.
        """
        # Configure optimizer with weight decay
        if self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")

        if not self.hparams.use_scheduler:
            return {"optimizer": optimizer}

        # Configure scheduler based on type
        if self.hparams.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif self.hparams.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif self.hparams.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

    def configure_gradient_clipping(
        self, 
        optimizer: torch.optim.Optimizer, 
        optimizer_idx: int, 
        gradient_clip_val: Optional[Union[int, float]] = None, 
        gradient_clip_algorithm: Optional[str] = None
    ) -> None:
        """Configure gradient clipping if specified."""
        if self.hparams.gradient_clip_val is not None:
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=self.hparams.gradient_clip_val, 
                gradient_clip_algorithm="norm"
            )

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Log learning rate
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        # Custom validation epoch end logic can be added here
        pass