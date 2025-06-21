"""
Advanced Model Evaluation and Reporting Tool for KrishiSahayak.

This module provides comprehensive metrics and utilities for model training and evaluation,
including accuracy calculations, loss tracking, and detailed performance reporting.
"""

import argparse
import json
import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall, ConfusionMatrix

# Import local modules
from src.models.hybrid import HybridModel
from src.data.dataset import PlantDiseaseDataset
from albumentations import Resize, Normalize, Compose
from albumentations.pytorch import ToTensorV2

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# Utility Classes
# =============================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name: str = '', fmt: str = ':f'):
        """
        Initialize the meter.
        
        Args:
            name: Name of the metric being tracked.
            fmt: Format string for string representation.
        """
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update the meter with a new value.
        
        Args:
            val: New value to include in the average.
            n: Number of samples this value represents (default: 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
    
    def __str__(self):
        """String representation of the meter."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# =============================================================================
# Core Metrics Functions
# =============================================================================

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    
    Args:
        output: Model output tensor of shape (batch_size, num_classes).
        target: Ground truth labels of shape (batch_size,).
        topk: Tuple of integers specifying the top-k accuracies to compute.
        
    Returns:
        List of accuracies for each k in topk.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res

def get_metrics(
    y_true: Union[torch.Tensor, np.ndarray, List[int]],
    y_pred: Union[torch.Tensor, np.ndarray, List[int]],
    y_prob: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
    average: str = 'macro',
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate various classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (for AUC-ROC and AUC-PR).
        average: Averaging method for multi-class metrics ('micro', 'macro', 'weighted', None).
        num_classes: Number of classes (required for some metrics).
        class_names: List of class names for per-class metrics.
        
    Returns:
        Dictionary containing various classification metrics.
    """
    # Convert inputs to numpy arrays if they're tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and torch.is_tensor(y_prob):
        y_prob = y_prob.cpu().numpy()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Add per-class metrics if class names are provided
    if class_names is not None and len(class_names) > 1:
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names, 
            output_dict=True, 
            zero_division=0
        )
        metrics['classification_report'] = report
        
        # Add per-class metrics
        for i, class_name in enumerate(class_names):
            metrics[f'precision_{class_name}'] = report[class_name]['precision']
            metrics[f'recall_{class_name}'] = report[class_name]['recall']
            metrics[f'f1_{class_name}'] = report[class_name]['f1-score']
    
    # Add confusion matrix
    if num_classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        metrics['confusion_matrix'] = cm.tolist()
    
    # Add AUC-ROC and AUC-PR if probabilities are provided
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 2:  # Multi-class
                metrics['auc_roc'] = roc_auc_score(
                    y_true, y_prob, 
                    multi_class='ovr', 
                    average=average
                )
                metrics['auc_pr'] = average_precision_score(
                    y_true, y_prob, 
                    average=average
                )
            else:  # Binary
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['auc_pr'] = average_precision_score(y_true, y_prob[:, 1])
        except Exception as e:
            logger.warning(f"Could not calculate AUC metrics: {e}")
    
    return metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_dataloader(config: dict) -> DataLoader:
    """Sets up and returns the test DataLoader based on config."""
    data_cfg = config['data']
    val_transform = Compose([
        Resize(height=data_cfg['image_size'][0], width=data_cfg['image_size'][1]),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    test_dataset = PlantDiseaseDataset(
        csv_path=data_cfg['csv_path'],
        data_dir=data_cfg['rgb_dir'],
        split='test',
        transform=val_transform,
        use_ms=config['model'].get('use_ms', False),
        ms_dir=data_cfg.get('ms_dir')
    )
    
    return DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=data_cfg.get('num_workers', 4),
        shuffle=False,
        pin_memory=True
    )

def evaluate_model(model: HybridModel, dataloader: DataLoader, device: torch.device, num_classes: int) -> dict:
    """Runs model evaluation and computes metrics."""
    model.to(device)
    model.eval()

    # Define metrics for overall and per-class performance
    metrics = MetricCollection({
        'accuracy': Accuracy(task="multiclass", num_classes=num_classes),
        'precision_macro': Precision(task="multiclass", num_classes=num_classes, average='macro'),
        'recall_macro': Recall(task="multiclass", num_classes=num_classes, average='macro'),
        'f1_macro': F1Score(task="multiclass", num_classes=num_classes, average='macro'),
        # Per-class metrics
        'f1_per_class': F1Score(task="multiclass", num_classes=num_classes, average='none'),
        'precision_per_class': Precision(task="multiclass", num_classes=num_classes, average='none'),
        'recall_per_class': Recall(task="multiclass", num_classes=num_classes, average='none'),
        # Confusion matrix
        'conf_matrix': ConfusionMatrix(task="multiclass", num_classes=num_classes),
    }).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating model"):
            # Move batch data to the target device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            preds = outputs.argmax(dim=1)
            targets = batch['label']
            
            metrics.update(preds, targets)
            
    # Compute final metrics from all batches
    final_metrics = metrics.compute()
    # Convert tensors to lists for JSON serialization
    return {k: v.cpu().tolist() for k, v in final_metrics.items()}

def generate_report(metrics: dict, class_names: list, output_dir: Path):
    """Generates and saves a JSON report and a confusion matrix plot."""
    logger.info("Generating evaluation report...")
    
    # --- Save detailed metrics to JSON ---
    report_path = output_dir / 'evaluation_report.json'
    # Create a more readable per-class report
    per_class_report = {}
    for i, class_name in enumerate(class_names):
        per_class_report[class_name] = {
            'f1_score': metrics['f1_per_class'][i],
            'precision': metrics['precision_per_class'][i],
            'recall': metrics['recall_per_class'][i],
        }
    
    full_report = {
        'overall_metrics': {
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
        },
        'per_class_metrics': per_class_report
    }
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=4)
    logger.info(f"Detailed metrics report saved to {report_path}")
    
    # --- Generate and save labeled confusion matrix plot ---
    cm_data = metrics['conf_matrix']
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        pd.DataFrame(cm_data, index=class_names, columns=class_names),
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plot_path = output_dir / 'confusion_matrix.png'
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model and generate detailed reports.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.ckpt) file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the project configuration YAML file.')
    parser.add_argument('--output-dir', type=str, help='Directory to save reports. Overrides config setting.')
    args = parser.parse_args()

    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir or config['project']['output_dir']) / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = setup_dataloader(config)
    
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = HybridModel.load_from_checkpoint(args.checkpoint)

    # --- Run Evaluation ---
    num_classes = config['model']['num_classes']
    computed_metrics = evaluate_model(model, dataloader, device, num_classes)
    
    # --- Generate Report ---
    class_names = config['data'].get('class_names', [f'Class_{i}' for i in range(num_classes)])
    generate_report(computed_metrics, class_names, output_dir)
    logger.info("Evaluation complete.")

if __name__ == '__main__':
    main()