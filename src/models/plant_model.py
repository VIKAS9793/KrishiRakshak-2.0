"""PyTorch Lightning model for plant disease classification."""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    _LRScheduler
)
from torchmetrics import (
    Accuracy, F1Score, Precision, Recall, 
    ConfusionMatrix, MeanSquaredError, 
    MeanAbsoluteError, StructuralSimilarityIndexMeasure,
    IoU, AveragePrecision,
    PeakSignalNoiseRatio, 
    SignalDistortionRatio
)
from torchvision import models
import torch.onnx
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import torch.quantization
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional


class PlantDiseaseModel(pl.LightningModule):
    """PyTorch Lightning model for plant disease classification."""
    
    def __init__(self, num_classes=38):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize MobileNetV3 Large
        self.base = models.mobilenet_v3_large(pretrained=True)
        
        # Freeze early layers for transfer learning
        for param in self.base.features[:10].parameters():
            param.requires_grad = False
        
        # Custom head for better feature extraction
        in_features = self.base.classifier[0].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Initialize comprehensive metrics for Phase 3."""
        # Core metrics
        self.train_metrics = MetricCollection({
            'accuracy': Accuracy(task='multiclass', num_classes=self.hparams.num_classes),
            'f1': F1Score(task='multiclass', num_classes=self.hparams.num_classes, average='weighted'),
            'precision': Precision(task='multiclass', num_classes=self.hparams.num_classes, average='weighted'),
            'recall': Recall(task='multiclass', num_classes=self.hparams.num_classes, average='weighted'),
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
            'ssim': StructuralSimilarityIndexMeasure(),
            'psnr': PeakSignalNoiseRatio(),
            'sdr': SignalDistortionRatio()
        }).clone(prefix='train_')
        
        self.val_metrics = self.train_metrics.clone(prefix='val_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')
        
        # Class-wise metrics
        self.class_metrics = {
            'accuracy': Accuracy(task='multiclass', num_classes=self.hparams.num_classes, average=None),
            'f1': F1Score(task='multiclass', num_classes=self.hparams.num_classes, average=None),
            'precision': Precision(task='multiclass', num_classes=self.hparams.num_classes, average=None),
            'recall': Recall(task='multiclass', num_classes=self.hparams.num_classes, average=None)
        }
        
        # Object detection metrics (for IoU and mAP)
        self.iou = IoU(num_classes=self.hparams.num_classes)
        self.map = AveragePrecision(num_classes=self.hparams.num_classes)
        
        # Additional metrics
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=self.hparams.num_classes)
        self.class_names = []  # Will be set during training
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        preds = torch.argmax(logits, dim=1)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_metrics(preds, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        
        # Log learning rate
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        preds = torch.argmax(logits, dim=1)
        self.log('val_loss', loss, prog_bar=True)
        self.val_metrics(preds, y)
        self.log_dict(self.val_metrics)
        
        return {'val_loss': loss, 'preds': preds, 'targets': y}
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        preds = torch.argmax(logits, dim=1)
        self.test_metrics(preds, y)
        self.confusion_matrix(preds, y)
        for metric in self.class_metrics.values():
            metric(preds, y)
        
        return {'test_loss': loss, 'preds': preds, 'targets': y}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        try:
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
            
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.epochs,
                eta_min=1e-6
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
        except Exception as e:
            logging.error(f"Optimizer configuration error: {str(e)}")
            raise
    
    def on_test_epoch_end(self):
        """Generate comprehensive test results for Phase 3 metrics."""
        try:
            # Calculate metrics
            metrics = self.test_metrics.compute()
            class_metrics = {name: metric.compute() for name, metric in self.class_metrics.items()}
            confusion_matrix = self.confusion_matrix.compute()
            
            # Calculate IoU and mAP
            iou = self.iou.compute()
            map = self.map.compute()
            
            # Log all metrics
            metrics.update({
                'iou': iou.item(),
                'map': map.item()
            })
            
            self.log_dict(metrics, on_epoch=True, prog_bar=True)
            
            # Generate visualization
            self._generate_visualizations(confusion_matrix, class_metrics)
            
            # Reset metrics
            self.test_metrics.reset()
            self.confusion_matrix.reset()
            self.iou.reset()
            self.map.reset()
            for metric in self.class_metrics.values():
                metric.reset()
            
        except Exception as e:
            logging.error(f"Error in test epoch end: {str(e)}")
            raise
    
    def _generate_visualizations(self, confusion_matrix, class_metrics):
        """Generate and save visualizations."""
        try:
            # Create output directory
            output_dir = Path('visualizations')
            output_dir.mkdir(exist_ok=True)
            
            # Plot confusion matrix
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                confusion_matrix.cpu().numpy(),
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(output_dir / 'confusion_matrix.png')
            plt.close()
            
            # Plot class-wise metrics
            for metric_name, values in class_metrics.items():
                plt.figure(figsize=(12, 6))
                plt.bar(self.class_names, values.cpu().numpy())
                plt.title(f'Class-wise {metric_name}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / f'class_{metric_name}.png')
                plt.close()
        except Exception as e:
            logging.error(f"Error generating visualizations: {str(e)}")
    
    def validate_model(self, datamodule) -> Dict[str, float]:
        """Validate the model before export."""
        try:
            # Set model to evaluation mode
            self.eval()
            
            # Get validation data
            val_loader = datamodule.val_dataloader()
            
            # Initialize metrics
            metrics = {
                'accuracy': Accuracy(task='multiclass', num_classes=self.hparams.num_classes),
                'f1': F1Score(task='multiclass', num_classes=self.hparams.num_classes, average='weighted'),
                'precision': Precision(task='multiclass', num_classes=self.hparams.num_classes, average='weighted'),
                'recall': Recall(task='multiclass', num_classes=self.hparams.num_classes, average='weighted')
            }
            
            # Validate
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    logits = self(x)
                    preds = torch.argmax(logits, dim=1)
                    
                    # Update metrics
                    for name, metric in metrics.items():
                        metric.update(preds, y)
            
            # Calculate final metrics
            results = {name: metric.compute().item() for name, metric in metrics.items()}
            
            # Log validation results
            logging.info(f"Validation metrics before export: {results}")
            
            return results
            
        except Exception as e:
            logging.error(f"Error during model validation: {str(e)}")
            raise
    
    def quantize_model(self) -> None:
        """Quantize the model to INT8."""
        try:
            # Set model to evaluation mode
            self.eval()
            
            # Fuse modules (if applicable)
            self.base = torch.quantization.fuse_modules(
                self.base,
                [['0', '1', '2']],  # Example fusion pattern
                inplace=True
            )
            
            # Prepare for quantization
            self.base.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Convert to quantized model
            self.base = torch.quantization.prepare(self.base)
            self.base = torch.quantization.convert(self.base)
            
            logging.info("Model quantized successfully to INT8")
            
        except Exception as e:
            logging.error(f"Error during model quantization: {str(e)}")
            raise
    
    def export_model(self, output_dir: str = 'models', quantize: bool = True) -> Dict[str, str]:
        """Export the model with multiple formats and optimizations."""
        try:
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # 0. Validate model before export
            logging.info("Validating model before export...")
            validation_results = self.validate_model(datamodule)
            
            # 1. Quantize model if requested
            if quantize:
                logging.info("Quantizing model...")
                self.quantize_model()
            
            # 2. Export to ONNX
            logging.info("Exporting to ONNX...")
            onnx_path = output_dir / 'model.onnx'
            dummy_input = torch.randn(1, 3, 224, 224)
            
            torch.onnx.export(
                self,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}}
            )
            
            # 3. Export to TorchScript
            logging.info("Exporting to TorchScript...")
            script_path = output_dir / 'model.pt'
            traced_model = torch.jit.trace(self, dummy_input)
            torch.jit.save(traced_model, script_path)
            
            # 4. Export to TFLite (with quantization)
            logging.info("Exporting to TFLite...")
            tflite_path = output_dir / 'model.tflite'
            converter = torch.jit.TFLiteConverter.from_torchscript(traced_model)
            tflite_model = converter.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # 5. Save class names
            logging.info("Saving class names...")
            class_names_path = output_dir / 'class_names.txt'
            with open(class_names_path, 'w') as f:
                f.write('\n'.join(self.class_names))
            
            # Return paths to exported files
            return {
                'onnx': str(onnx_path),
                'torchscript': str(script_path),
                'tflite': str(tflite_path),
                'class_names': str(class_names_path)
            }
            
        except Exception as e:
            logging.error(f"Error exporting model: {str(e)}")
            raise
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Prediction step for inference."""
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        return self(x)
