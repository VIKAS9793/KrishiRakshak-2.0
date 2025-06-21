"""
Utility functions for the training pipeline.
"""

import importlib
import json
import logging
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Type variable for optimizer and scheduler
T = TypeVar('T', bound=Optimizer)


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: str = 'INFO',
    console: bool = True,
    file_mode: str = 'w',
    format_str: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to the log file. If None, no file logging is done.
        level: Logging level (e.g., 'INFO', 'DEBUG', 'WARNING').
        console: Whether to log to console.
        file_mode: File mode for the log file ('w' for write, 'a' for append).
        format_str: Logging format string.
    """
    # Convert string level to logging level
    level = getattr(logging, level.upper())
    
    # Clear any existing handlers
    logging.root.handlers = []
    
    # Configure root logger
    logging.basicConfig(level=level, format=format_str)
    
    # Remove default handler if console logging is disabled
    if not console:
        logging.root.handlers = []
    
    # Add file handler if log_file is provided
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(level)
        formatter = logging.Formatter(format_str)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
    
    # Set NumPy and PyTorch logging levels
    np_log_level = 'WARNING' if level > logging.DEBUG else 'INFO'
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
    os.environ['KMP_WARNINGS'] = '0'  # Suppress OpenMP warnings
    
    # Suppress PIL logging
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.WARNING)
    
    # Suppress matplotlib logging
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized at level {logging.getLevelName(level)}")


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        OmegaConf DictConfig object with the configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load base config
    config = OmegaConf.load(config_path)
    
    # Resolve any interpolation references
    OmegaConf.resolve(config)
    
    return config


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration to save.
        output_path: Path to save the configuration file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        OmegaConf.save(config=config, f=f)
    
    logger.info(f"Configuration saved to {output_path}")


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Set random seed to {seed}")


def get_optimizer(
    model: nn.Module,
    optimizer_config: DictConfig
) -> Optimizer:
    """
    Get optimizer based on configuration.
    
    Args:
        model: Model whose parameters to optimize.
        optimizer_config: Optimizer configuration.
        
    Returns:
        Optimizer instance.
    """
    optimizer_name = optimizer_config.name.lower()
    lr = optimizer_config.learning_rate
    weight_decay = optimizer_config.get('weight_decay', 0.0)
    
    # Filter parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name == 'adam':
        return optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **optimizer_config.get('kwargs', {})
        )
    elif optimizer_name == 'adamw':
        return optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **optimizer_config.get('kwargs', {})
        )
    elif optimizer_name == 'sgd':
        return optim.SGD(
            params,
            lr=lr,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=optimizer_config.get('nesterov', True),
            **optimizer_config.get('kwargs', {})
        )
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=optimizer_config.get('momentum', 0.9),
            **optimizer_config.get('kwargs', {})
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: Optimizer,
    scheduler_config: DictConfig,
    steps_per_epoch: Optional[int] = None
) -> Optional[lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule.
        scheduler_config: Scheduler configuration.
        steps_per_epoch: Number of steps per epoch (required for some schedulers).
        
    Returns:
        Learning rate scheduler or None if no scheduler is configured.
    """
    if not scheduler_config or not scheduler_config.enabled:
        return None
    
    scheduler_name = scheduler_config.name.lower()
    
    if scheduler_name == 'steplr':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.step_size,
            gamma=scheduler_config.gamma,
            **scheduler_config.get('kwargs', {})
        )
    elif scheduler_name == 'multisteplr':
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config.milestones,
            gamma=scheduler_config.gamma,
            **scheduler_config.get('kwargs', {})
        )
    elif scheduler_name == 'cosineannealinglr':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('t_max', 10),
            eta_min=scheduler_config.get('eta_min', 0),
            **scheduler_config.get('kwargs', {})
        )
    elif scheduler_name == 'reduceonplateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10),
            verbose=True,
            **scheduler_config.get('kwargs', {})
        )
    elif scheduler_name == 'onecyclelr':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required for OneCycleLR scheduler")
        
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_config.max_lr,
            epochs=scheduler_config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=scheduler_config.get('pct_start', 0.3),
            anneal_strategy=scheduler_config.get('anneal_strategy', 'cos'),
            **scheduler_config.get('kwargs', {})
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Union[str, torch.device] = 'cuda'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        device: Device to load the model to.
        
    Returns:
        Dictionary with loaded state and additional information.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Extract additional information
    result = {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', float('inf')),
        'config': checkpoint.get('config', {}),
        'metrics': checkpoint.get('metrics', {})
    }
    
    logger.info(f"Loaded checkpoint from epoch {result['epoch']}")
    return result


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    epoch: int,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    best_metric: Optional[float] = None,
    config: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save.
        checkpoint_path: Path to save the checkpoint.
        epoch: Current epoch.
        optimizer: Optimizer state to save.
        scheduler: Scheduler state to save.
        best_metric: Best metric value so far.
        config: Configuration to save.
        metrics: Training metrics to save.
        is_best: Whether this is the best model so far.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare state dicts
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_metric': best_metric,
        'config': config or {},
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save as best model if specified
    if is_best:
        best_path = checkpoint_path.parent / 'best_model.pt'
        shutil.copyfile(checkpoint_path, best_path)
        logger.info(f"Saved best model to {best_path}")


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_id: int = 0) -> torch.device:
    """
    Get the appropriate device (CPU or GPU).
    
    Args:
        device_id: GPU device ID to use.
        
    Returns:
        Device object.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')
        logger.info(f'Using GPU: {torch.cuda.get_device_name(device_id)}')
    else:
        device = torch.device('cpu')
        logger.info('Using CPU')
    
    return device


def freeze_model(
    model: nn.Module,
    freeze: bool = True,
    exclude: Optional[List[str]] = None
) -> None:
    """
    Freeze or unfreeze model parameters.
    
    Args:
        model: Model to modify.
        freeze: Whether to freeze parameters (True) or unfreeze (False).
        exclude: List of parameter names to exclude from freezing.
    """
    exclude = exclude or []
    
    for name, param in model.named_parameters():
        if not any(ex in name for ex in exclude):
            param.requires_grad = not freeze
    
    logger.info(f"Model parameters {'frozen' if freeze else 'unfrozen'}")


def get_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute gradient norm of model parameters.
    
    Args:
        model: Model to compute gradient norm for.
        norm_type: Type of norm to compute (e.g., 2 for L2 norm).
        
    Returns:
        Gradient norm.
    """
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        return 0.0
    
    device = parameters[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type
    )
    
    return total_norm.item()


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Applies MixUp augmentation to the input batch.
    
    Args:
        x: Input batch of shape (batch_size, ...).
        y: Target labels of shape (batch_size,).
        alpha: MixUp alpha parameter.
        
    Returns:
        Mixed inputs, shuffled inputs, lambda, and shuffled targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Applies CutMix augmentation to the input batch.
    
    Args:
        x: Input batch of shape (batch_size, C, H, W).
        y: Target labels of shape (batch_size,).
        alpha: CutMix alpha parameter.
        
    Returns:
        Mixed inputs, shuffled inputs, lambda, and shuffled targets.
    """
    if alpha <= 0:
        return x, y, 1.0, y
    
    # Generate mixed sample
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size(0), device=x.device)
    target_a = y
    target_b = y[rand_index]
    
    # Get the bounding box coordinates
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    # Apply CutMix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, target_a, target_b, lam


def rand_bbox(
    size: Tuple[int, ...],
    lam: float
) -> Tuple[int, int, int, int]:
    """
    Generate random bounding box for CutMix.
    
    Args:
        size: Input size (batch_size, C, H, W).
        lam: Lambda value from beta distribution.
        
    Returns:
        Bounding box coordinates (x1, y1, x2, y2).
    """
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


def get_model_summary(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: Union[str, torch.device] = 'cuda'
) -> str:
    """
    Get a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model.
        input_size: Input size (C, H, W).
        device: Device to run the model on.
        
    Returns:
        Model summary as a string.
    """
    try:
        from torchsummary import summary
        model_summary = []
        
        def hook(module, input, output):
            class_name = module.__class__.__name__
            module_idx = len(model_summary)
            
            # Get input and output shapes
            if isinstance(input, (tuple, list)) and input:
                input_shape = [list(inp.size()) for inp in input if torch.is_tensor(inp)]
            else:
                input_shape = []
                
            if isinstance(output, (tuple, list)) and output:
                output_shape = [list(out.size()) for out in output if torch.is_tensor(out)]
            else:
                output_shape = []
            
            # Count parameters
            params = 0
            if hasattr(module, 'weight') and hasattr(module.weight, 'size') and module.weight.requires_grad:
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if hasattr(module, 'bias') and hasattr(module.bias, 'size') and module.bias is not None and module.bias.requires_grad:
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
            
            model_summary.append({
                'layer_type': class_name,
                'input_shape': input_shape,
                'output_shape': output_shape,
                'params': params,
                'trainable': any(p.requires_grad for p in module.parameters())
            })
        
        # Register hooks
        hooks = []
        for layer in model.modules():
            hook = layer.register_forward_hook(hook)
            hooks.append(hook)
        
        # Run forward pass
        dummy_input = torch.randn(1, *input_size).to(device)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Format summary
        summary_str = f"{'Layer (type)':<40} {'Output Shape':<30} {'Param #'}"
        summary_str += "\n" + "=" * 80 + "\n"
        
        total_params = 0
        total_output = 0
        trainable_params = 0
        
        for layer in model_summary:
            # Input shape
            line_new = f"{layer['layer_type']}"
            
            # Output shape
            if layer['output_shape']:
                line_new += f"{' ' * (40 - len(line_new))} {str(layer['output_shape']):<30}"
            else:
                line_new += " " * (71 - len(line_new))
            
            # Parameters
            line_new += f"{layer['params']:,}"
            
            # Add trainable indicator
            if not layer['trainable']:
                line_new += " (frozen)"
            
            summary_str += line_new + "\n"
            
            total_params += layer['params']
            if layer['output_shape']:
                total_output += sum(np.prod(s) for s in layer['output_shape'])
            if layer['trainable']:
                trainable_params += layer['params']
        
        # Add total parameters
        summary_str += "=" * 80 + "\n"
        summary_str += f"Total params: {total_params:,}\n"
        summary_str += f"Trainable params: {trainable_params:,}\n"
        summary_str += f"Non-trainable params: {total_params - trainable_params:,}\n"
        
        # Add input size and forward/backward passes
        summary_str += "-" * 80 + "\n"
        summary_str += f"Input size (MB): {np.prod(input_size) * 4 / (1024 * 1024):.2f}\n"
        summary_str += f"Forward/backward pass size (MB): {total_output * 4 / (1024 * 1024):.2f}\n"
        
        return summary_str
        
    except ImportError:
        return "torchsummary not installed. Install with: pip install torchsummary"
    except Exception as e:
        return f"Error generating model summary: {str(e)}"
