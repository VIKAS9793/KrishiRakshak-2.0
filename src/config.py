"""
Modern ML/DL Configuration for KrishiRakshak
Following 2024-2025 state-of-the-art practices
"""
import os
import warnings
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, TypeVar

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    OneCycleLR, 
    CosineAnnealingWarmRestarts, 
    _LRScheduler,
    ReduceLROnPlateau
)
import timm

# Modern augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose as AlbumentationsCompose

# Metrics
from torchmetrics import MetricCollection, Accuracy, F1Score, Precision, Recall, AUROC


class Config:
    # ========== System Configuration ==========
    # Random seed for reproducibility
    SEED = 42
    DETERMINISTIC = True  # Enable deterministic algorithms
    BENCHMARK = False  # Disable cudnn benchmark for reproducibility
    
    # Hardware detection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Modern training settings
    USE_AMP = True  # Automatic Mixed Precision
    COMPILE_MODEL = True  # PyTorch 2.0+ compilation
    COMPILE_MODE = "default"  # Options: "default", "reduce-overhead", "max-autotune"
    
    # ========== Data Loading ==========
    # Hardware-specific settings
    if DEVICE == "cuda":
        # GPU-optimized settings
        BATCH_SIZE = 64
        NUM_WORKERS = 4
        PIN_MEMORY = True
        PREFETCH_FACTOR = 2
    else:
        # CPU-optimized settings
        BATCH_SIZE = 16
        NUM_WORKERS = 2
        PIN_MEMORY = False
        PREFETCH_FACTOR = 1
    
    # Advanced data loading
    PERSISTENT_WORKERS = True  # Modern DataLoader optimization
    MULTIPROCESSING_CONTEXT = "spawn"  # Better for CUDA
    CACHE_IMAGES = True  # Set False if memory is limited
    MIN_SAMPLES_PER_CLASS = 5  # Minimum samples required per class for training
        
    # ========== Training Parameters ==========
    # Training loop
    NUM_EPOCHS = 50
    MAX_EPOCHS = 100
    MIN_EPOCHS = 10
    PATIENCE = 5  # For early stopping
    EARLY_STOPPING_PATIENCE = 10
    MIN_DELTA = 1e-4
    
    # Optimizer
    OPTIMIZER = "adamw"
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    
    # Scheduler
    SCHEDULER = "onecycle"  # Options: "onecycle", "cosine", "reduce_on_plateau"
    MAX_LR = 3e-3
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 5
    LR_PATIENCE = 5  # For reduce_on_plateau
    LR_FACTOR = 0.1  # For reduce_on_plateau
    
    # Regularization
    DROPOUT_RATE = 0.2
    DROP_PATH_RATE = 0.1
    LABEL_SMOOTHING = 0.1
    GRADIENT_CLIP_VAL = 1.0
    
    # Advanced training
    ACCUMULATE_GRAD_BATCHES = 1
    PRECISION = 16 if USE_AMP else 32
    
    # ========== Data Augmentation (Modern 2024) ==========
    # Image parameters
    IMG_SIZE = 224  # Input image size (height=width)
    MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    STD = [0.229, 0.224, 0.225]   # ImageNet std
    
    # Modern augmentation probabilities
    AUGMENTATION_PROB = 0.8  # Higher for more aggressive augmentation
    
    # Geometric augmentations
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB = 0.2
    ROTATION_RANGE = 45  # degrees
    SHIFT_LIMIT = 0.15  # 15% of image size
    ZOOM_RANGE = [0.8, 1.2]  # Random zoom range
    
    # Advanced color jitter
    BRIGHTNESS_LIMIT = 0.3
    CONTRAST_LIMIT = 0.3
    SATURATION_LIMIT = 0.3
    HUE_LIMIT = 0.15
    
    # Modern augmentation techniques
    RANDOM_ERASING_PROB = 0.2
    RANDOM_ERASING_SCALE = (0.02, 0.33)
    RANDOM_ERASING_RATIO = (0.3, 3.3)
    
    # Cutout/Erasing (modern implementation)
    CUTOUT_ENABLED = True
    CUTOUT_NUM_HOLES = 8
    CUTOUT_MAX_H_SIZE = 32
    CUTOUT_MAX_W_SIZE = 32
    CUTOUT_FILL_VALUE = 0
    
    # Advanced augmentation settings
    USE_AUTOAUGMENT = True
    USE_RANDOM_ERASING = True
    USE_GRID_MASK = False  # GridMask augmentation
    
    # Test-time augmentation
    TTA_NUM_AUGS = 5  # Number of augmentations per image
    TTA_MERGE = 'mean'  # How to combine TTA predictions: 'mean' or 'gmean'
    
    # Mixup/Cutmix (modern regularization)
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.5
    
    # ========== Model Architecture ==========
    MODEL_NAME = "efficientnetv2_rw_s"  # Modern efficient architecture
    PRETRAINED = True
    NUM_CLASSES = 38
    
    # Transfer learning
    FREEZE_BACKBONE = True
    UNFREEZE_EPOCH = 5
    UNFREEZE_GROUPS = 5
    
    # ========== Metrics ==========
    METRICS = {
        'accuracy': Accuracy(task='multiclass', num_classes=38),
        'f1_macro': F1Score(task='multiclass', num_classes=38, average='macro'),
        'f1_weighted': F1Score(task='multiclass', num_classes=38, average='weighted'),
        'precision': Precision(task='multiclass', num_classes=38, average='macro'),
        'recall': Recall(task='multiclass', num_classes=38, average='macro'),
        'auroc': AUROC(task='multiclass', num_classes=38)
    }
    
    # ========== Project Structure ==========
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    FIXED_DATA_DIR = DATA_DIR / "fixed"
    MODEL_DIR = ROOT_DIR / "models"
    LOGS_DIR = ROOT_DIR / "logs"
    REPORTS_DIR = ROOT_DIR / "reports"
    SRC_DIR = ROOT_DIR / "src"
    
    @classmethod
    def setup_directories(cls):
        """Create all necessary project directories."""
        for dir_path in [
            cls.MODEL_DIR, 
            cls.LOGS_DIR, 
            cls.REPORTS_DIR, 
            cls.FIXED_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.DATA_DIR,
            cls.CHECKPOINT_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # ========== Logging and Checkpoints ==========
    CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
    CHECKPOINT_SAVE_TOP_K = 3
    MONITOR_METRIC = "val_loss"
    MODE = "min"
    LOGGER = "tensorboard"
    LOG_EVERY_N_STEPS = 10
    @classmethod
    def log_hardware_info(cls):
        """Log detailed hardware and configuration information."""
        import platform
        import psutil
        
        # Basic system info
        info = [
            f"System: {platform.system()} {platform.release()}",
            f"Processor: {platform.processor()}",
            f"Python: {platform.python_version()}",
            f"PyTorch: {torch.__version__}",
            f"CUDA Available: {torch.cuda.is_available()}",
            f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}",
            f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}",
            f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical",
            f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB",
            "",
            "=== Training Configuration ===",
            f"Device: {cls.DEVICE.upper()}",
            f"Precision: {cls.PRECISION}-bit{' (AMP)' if cls.USE_AMP else ''}",
            f"Model: {cls.MODEL_NAME} (pretrained: {cls.PRETRAINED})",
            f"Optimizer: {cls.OPTIMIZER} (lr: {cls.LEARNING_RATE}, wd: {cls.WEIGHT_DECAY})",
            f"Scheduler: {cls.SCHEDULER}",
            f"Batch Size: {cls.BATCH_SIZE} (accumulate: {cls.ACCUMULATE_GRAD_BATCHES})",
            f"Epochs: {cls.NUM_EPOCHS} (early stop: {cls.EARLY_STOPPING_PATIENCE})",
            "",
            "=== Augmentation ===",
            f"Image Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}",
            f"Augmentation: {cls.AUGMENTATION_PROB*100}% strength",
            f"Mixup: {cls.MIXUP_ALPHA if cls.MIXUP_ALPHA > 0 else 'Off'}",
            f"CutMix: {cls.CUTMIX_ALPHA if cls.CUTMIX_ALPHA > 0 else 'Off'}",
            f"TTA: {cls.TTA_NUM_AUGS} augs"
        ]
        
        print("\n".join(["="*50, "KrishiRakshak Configuration", "="*50] + info + ["="*50]))
    
    @classmethod
    def get_optimizer(cls, model: nn.Module) -> Optimizer:
        """Get configured optimizer.
        
        Args:
            model: The model whose parameters will be optimized
            
        Returns:
            Configured optimizer
            
        Raises:
            ValueError: If the specified optimizer is not supported
        """
        if cls.OPTIMIZER.lower() == "adamw":
            return AdamW(
                model.parameters(),
                lr=cls.LEARNING_RATE,
                weight_decay=cls.WEIGHT_DECAY,
                betas=cls.BETAS,
                eps=cls.EPS
            )
        raise ValueError(f"Unsupported optimizer: {cls.OPTIMIZER}")
    
    @classmethod
    def get_scheduler(
        cls, 
        optimizer: Optimizer, 
        steps_per_epoch: int
    ) -> Optional[Union[OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau]]:
        """Get learning rate scheduler.
        
        Args:
            optimizer: The optimizer to wrap with the scheduler
            steps_per_epoch: Number of steps per training epoch
            
        Returns:
            Configured learning rate scheduler or None if no scheduler is specified
        """
        if cls.SCHEDULER == "onecycle":
            return OneCycleLR(
                optimizer,
                max_lr=cls.LEARNING_RATE,
                epochs=cls.NUM_EPOCHS,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1e4,
                anneal_strategy='cos'
            )
        elif cls.SCHEDULER == "cosine":
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=1,
                eta_min=cls.MIN_LR
            )
        elif cls.SCHEDULER == "reduce_on_plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=cls.LR_FACTOR,
                patience=cls.LR_PATIENCE,
                min_lr=cls.MIN_LR
            )
        return None
    
    # Data augmentation
    @classmethod
    def get_train_transforms(cls):
        """Get modern training data transformation pipeline with 2024 best practice augmentations."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Base augmentations
        transforms = [
            # Modern resize strategy with random crop
            A.RandomResizedCrop(
                height=cls.IMG_SIZE,
                width=cls.IMG_SIZE,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                interpolation=1,  # Bilinear
                p=1.0
            ),
            
            # Geometric augmentations
            A.HorizontalFlip(p=cls.HORIZONTAL_FLIP_PROB),
            A.VerticalFlip(p=cls.VERTICAL_FLIP_PROB),
            A.ShiftScaleRotate(
                shift_limit=cls.SHIFT_LIMIT,
                scale_limit=0.2,
                rotate_limit=cls.ROTATION_RANGE,
                interpolation=1,
                border_mode=0,
                p=0.5
            ),
        ]
        
        # Modern color augmentations
        if cls.USE_AUTOAUGMENT:
            transforms.extend([
                A.OneOf([
                    A.ColorJitter(
                        brightness=cls.BRIGHTNESS_LIMIT,
                        contrast=cls.CONTRAST_LIMIT,
                        saturation=cls.SATURATION_LIMIT,
                        hue=cls.HUE_LIMIT,
                        p=0.8
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.8
                    ),
                ], p=0.8),
            ])
        
        # Advanced augmentations
        transforms.extend([
            # Cutout/CoarseDropout
            A.CoarseDropout(
                max_holes=cls.CUTOUT_NUM_HOLES,
                max_height=cls.CUTOUT_MAX_H_SIZE,
                max_width=cls.CUTOUT_MAX_W_SIZE,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=cls.CUTOUT_FILL_VALUE,
                p=0.5 if cls.CUTOUT_ENABLED else 0.0
            ),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=5, p=0.3),
            ], p=0.3),
            
            # GridMask (modern augmentation)
            A.GridDropout(
                ratio=0.3,
                unit_size_min=10,
                unit_size_max=30,
                random_offset=True,
                p=0.3 if cls.USE_GRID_MASK else 0.0
            ),
        ])
        
        # Final transforms
        transforms.extend([
            A.Normalize(
                mean=cls.MEAN,
                std=cls.STD,
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(p=1.0),
        ])
        
        return A.Compose(transforms, p=cls.AUGMENTATION_PROB)
        
    @classmethod
    def _get_base_transforms(cls, tta: bool = False):
        """Get base transforms used for validation and test."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        transforms = [
            A.Resize(
                height=cls.IMG_SIZE,
                width=cls.IMG_SIZE,
                interpolation=1,  # Bilinear
                p=1.0
            )
        ]
        
        if not tta:
            transforms.extend([
                A.Normalize(
                    mean=cls.MEAN,
                    std=cls.STD,
                    max_pixel_value=255.0,
                    p=1.0
                ),
                ToTensorV2(p=1.0)
            ])
        
        return transforms
    
    @classmethod
    def get_val_transforms(cls) -> A.Compose:
        """Get validation data transformation pipeline.
        
        Returns:
            Composed validation transforms
        """
        transforms = cls._get_base_transforms()
        return A.Compose(transforms, p=1.0)
    
    @classmethod
    def get_tta_transforms(cls) -> List[Any]:
        """Get test-time augmentation transforms.
        
        Returns:
            List of composed transforms for test-time augmentation
        """
        base_transforms = cls._get_base_transforms(tta=True)
        
        # Define TTA transforms
        tta_transforms = [
            A.Compose([  # Original
                *base_transforms,
                A.Normalize(mean=cls.MEAN, std=cls.STD, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0)
            ], p=1.0),
            A.Compose([  # Horizontal flip
                A.HorizontalFlip(p=1.0),
                *base_transforms,
                A.Normalize(mean=cls.MEAN, std=cls.STD, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0)
            ], p=1.0),
            A.Compose([  # Vertical flip
                A.VerticalFlip(p=1.0),
                *base_transforms,
                A.Normalize(mean=cls.MEAN, std=cls.STD, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0)
            ], p=1.0)
        ]
        
        return tta_transforms[:cls.TTA_NUM_AUGS] if cls.TTA_NUM_AUGS > 1 else tta_transforms[:1]
        
    @classmethod
    def get_test_transforms(cls) -> Union[A.Compose, List[A.Compose]]:
        """Get test data transformation pipeline with TTA support.
        
        Returns:
            Either a single Compose transform or a list of transforms for TTA
        """
        if cls.TTA_NUM_AUGS > 1:
            return cls.get_tta_transforms()
        return cls.get_val_transforms()
    
    @classmethod
    def get_hardware_info(cls) -> Dict[str, str]:
        """Get hardware information."""
        import platform
        import psutil
        
        info = {
            "system": f"{platform.system()} {platform.release()}",
            "processor": platform.processor(),
            "physical_cores": str(psutil.cpu_count(logical=False)),
            "total_cores": str(psutil.cpu_count(logical=True)),
            "total_ram_gb": f"{psutil.virtual_memory().total / (1024**3):.1f}",
            "device": cls.DEVICE,
        }
        
        if cls.DEVICE == "cuda":
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        
        return info
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {}
        for key in dir(cls):
            if key.isupper() and not key.startswith('_'):
                config_dict[key] = getattr(cls, key)
        return config_dict
        
    @classmethod
    def update_from_file(cls, file_path: Union[str, Path]) -> None:
        """Update class attributes from a YAML file.
        
        Args:
            file_path: Path to the YAML config file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            ValueError: If config contains invalid keys
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            try:
                data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Invalid YAML in config file: {e}")
        
        # Only update existing attributes
        for key, value in data.items():
            key_upper = key.upper()
            if hasattr(cls, key_upper):
                setattr(cls, key_upper, value)
            else:
                warnings.warn(f"Ignoring unknown config key: {key}", UserWarning)
    
    @classmethod
    def setup_training(
        cls, 
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader
    ) -> tuple[
        nn.Module, 
        Optimizer, 
        Optional[Union[OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau]], 
        Optional[torch.cuda.amp.GradScaler]
    ]:
        """Setup training components.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            
        Returns:
            Tuple of (model, optimizer, scheduler, scaler)
        """
        # Set random seeds
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
        
        # Configure deterministic algorithms if needed
        if cls.DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Setup optimizer and scheduler
        optimizer = cls.get_optimizer(model)
        scheduler = cls.get_scheduler(optimizer, len(train_loader)) if cls.SCHEDULER else None
        
        # Setup mixed precision
        scaler = torch.cuda.amp.GradScaler() if cls.USE_AMP and torch.cuda.is_available() else None
        
        # Compile model if enabled (PyTorch 2.0+)
        if cls.COMPILE_MODEL and hasattr(torch, 'compile'):
            model = torch.compile(model, mode=cls.COMPILE_MODE)
        
        return model, optimizer, scheduler, scaler

# Directory setup should be called explicitly from the main script
# Example: Config.setup_directories()
