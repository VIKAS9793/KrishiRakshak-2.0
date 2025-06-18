import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, MetricCollection, Precision, Recall
from torchvision import models

# Configure a centralized logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class PlantDiseaseModel(pl.LightningModule):
    """
    A robust PyTorch Lightning model for plant disease classification.

    This model uses a pre-trained MobileNetV3 Large backbone with a custom
    classifier head. It is designed for efficient training and includes
    comprehensive, appropriate metrics for classification tasks.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_name: str = 'cosine',
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self._freeze_layers()
        self._create_classifier_head()
        
        self.criterion = nn.CrossEntropyLoss()
        self._setup_metrics()

    def _freeze_layers(self):
        """Freezes the first 10 feature blocks for transfer learning."""
        for param in self.base.features[:10].parameters():
            param.requires_grad = False

    def _create_classifier_head(self):
        """Replaces the final layer with a custom classifier head."""
        in_features = self.base.classifier[0].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, self.hparams.num_classes)
        )

    def _setup_metrics(self):
        """Initializes core classification metrics from torchmetrics."""
        metrics = {
            'accuracy': Accuracy(task='multiclass', num_classes=self.hparams.num_classes),
            'f1_weighted': F1Score(task='multiclass', num_classes=self.hparams.num_classes, average='weighted'),
            'precision_weighted': Precision(task='multiclass', num_classes=self.hparams.num_classes, average='weighted'),
            'recall_weighted': Recall(task='multiclass', num_classes=self.hparams.num_classes, average='weighted'),
        }
        self.train_metrics = MetricCollection(metrics).clone(prefix='train_')
        self.val_metrics = MetricCollection(metrics).clone(prefix='val_')
        self.test_metrics = MetricCollection(metrics).clone(prefix='test_')
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def _common_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """A common step for training, validation, and testing."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, y = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_metrics.update(preds, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metrics.update(preds, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, preds, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.test_metrics.update(preds, y)
        self.confusion_matrix.update(preds, y)

    def on_test_epoch_end(self):
        """Generates and logs test results at the end of the test phase."""
        log.info("Test epoch finished. Computing and logging final metrics.")
        try:
            test_metrics_computed = self.test_metrics.compute()
            self.log_dict(test_metrics_computed)
            log.info(f"Final Test Metrics: {test_metrics_computed}")

            confusion_matrix_computed = self.confusion_matrix.compute()
            self._generate_confusion_matrix_viz(confusion_matrix_computed)
        except Exception as e:
            log.error(f"Error during on_test_epoch_end: {e}", exc_info=True)
        finally:
            self.test_metrics.reset()
            self.confusion_matrix.reset()

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler_name = self.hparams.scheduler_name.lower()
        if scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        elif scheduler_name == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=self.hparams.scheduler_factor,
                patience=self.hparams.scheduler_patience, verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.hparams.scheduler_name}")

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def _generate_confusion_matrix_viz(self, conf_matrix: torch.Tensor):
        """Generates and saves a confusion matrix plot."""
        try:
            output_dir = Path(self.trainer.logger.log_dir or 'visualizations')
            output_dir.mkdir(exist_ok=True, parents=True)
            
            fig, ax = plt.subplots(figsize=(18, 15))
            sns.heatmap(
                conf_matrix.cpu().numpy(), annot=True, fmt='d', cmap='Blues',
                xticklabels=self.hparams.class_names, yticklabels=self.hparams.class_names, ax=ax
            )
            ax.set_title('Confusion Matrix', fontsize=16)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            save_path = output_dir / 'confusion_matrix.png'
            plt.savefig(save_path)
            plt.close(fig)
            log.info(f"Confusion matrix saved to {save_path}")
        except Exception as e:
            log.error(f"Error generating confusion matrix visualization: {e}", exc_info=True)

    def _validate_input_sample(self, input_sample: torch.Tensor):
        """Validates the shape of the input sample for export."""
        expected_shape = (1, 3, 224, 224)
        if input_sample.shape != expected_shape:
            raise ValueError(
                f"Invalid input_sample shape. Expected {expected_shape}, but got {input_sample.shape}"
            )

    def _save_pytorch_model(self, output_path: Path):
        """Helper function to save the model's state_dict for web UI use."""
        pytorch_model_path = output_path / 'model.pth'
        try:
            torch.save(self.state_dict(), str(pytorch_model_path))
            log.info(f"Model state_dict successfully saved to PyTorch format: {pytorch_model_path}")
        except Exception as e:
            log.error(f"Failed to save PyTorch model: {e}", exc_info=True)

    def _export_to_onnx(self, output_path: Path, input_sample: torch.Tensor):
        """Helper function to export the model to ONNX format for mobile deployment."""
        onnx_path = output_path / 'model.onnx'
        try:
            self._validate_input_sample(input_sample)
            torch.onnx.export(
                self, input_sample, str(onnx_path), export_params=True, opset_version=12,
                do_constant_folding=True, input_names=['input'], output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            log.info(f"Model successfully exported to ONNX: {onnx_path}")
        except Exception as e:
            log.error(f"Failed to export to ONNX: {e}", exc_info=True)

    def _save_class_names(self, output_path: Path):
        """Helper function to save class names to a text file."""
        class_names_path = output_path / 'class_names.txt'
        try:
            with open(class_names_path, 'w') as f:
                f.write('\n'.join(self.hparams.class_names))
            log.info(f"Class names saved to {class_names_path}")
        except Exception as e:
            log.error(f"Failed to save class names: {e}", exc_info=True)

    def export_model(self, output_dir: str = 'models', input_sample: Optional[torch.Tensor] = None):
        """
        Exports the trained model to multiple deployable formats.

        - .pth: Standard PyTorch state_dict for use in Python environments (e.g., web UI).
        - .onnx: Standard format for mobile deployment and cross-platform compatibility.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        log.info(f"Exporting model to {output_path}...")

        self.eval()
        
        # Save PyTorch model state_dict for web backend
        self._save_pytorch_model(output_path)
        
        # Prepare for and export to ONNX for mobile
        sample = input_sample if input_sample is not None else torch.randn(1, 3, 224, 224, device=self.device)
        self._export_to_onnx(output_path, sample)
        
        # Save artifacts needed for inference
        self._save_class_names(output_path)
        self.advise_on_quantization()

    def advise_on_quantization(self):
        """Provides guidance on how to correctly perform post-training quantization."""
        log.warning("--- Model Quantization Advisory ---")
        log.warning("Post-training static quantization requires a careful calibration process with representative data.")
        log.info("Recommended Steps (in a separate script):")
        log.info("1. Load trained model. 2. `model.eval()`. 3. Fuse modules.")
        log.info("4. Prepare model for quantization. 5. Calibrate with validation data.")
        log.info("6. Convert to quantized model. 7. Save and deploy.")
        log.warning("------------------------------------")