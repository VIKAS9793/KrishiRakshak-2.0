"""
Modern ML/DL Configuration for KrishiRakshak
Following 2024-2025 state-of-the-art practices
"""
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import timm

# Modern augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    
    # Advanced data loading
    PERSISTENT_WORKERS = True
    MULTIPROCESSING_CONTEXT = "spawn"  # Better for CUDA
    CACHE_IMAGES = True  # Set False if memory is limited
    
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
    
    # Image caching (set to False if memory is limited)
    CACHE_IMAGES = True
        
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
    SCHEDULER = "onecycle"
    LR_SCHEDULER = "reduce_on_plateau"
    MAX_LR = 3e-3
    MIN_LR = 1e-6
    LR_FACTOR = 0.1
    LR_PATIENCE = 5
    WARMUP_EPOCHS = 5
    
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
    
    # Create required directories
    for dir_path in [
        MODEL_DIR, 
        LOGS_DIR, 
        REPORTS_DIR, 
        FIXED_DATA_DIR,
        PROCESSED_DATA_DIR,
        DATA_DIR
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ========== Logging and Checkpoints ==========
    CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
    CHECKPOINT_SAVE_TOP_K = 3
    MONITOR_METRIC = "val_loss"
    MODE = "min"
    
    # Logging
    LOGGER = "tensorboard"
    LOG_EVERY_N_STEPS = 10
    
    # Experiment
    EXPERIMENT_NAME = "krishirakshak"  # For experiment tracking
    
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
    def get_optimizer(cls, model: nn.Module) -> torch.optim.Optimizer:
        """Get configured optimizer."""
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
    def get_scheduler(cls, optimizer, steps_per_epoch: int):
        """Get learning rate scheduler."""
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
    def get_val_transforms(cls):
        """Get validation data transformation pipeline with TTA support."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        return A.Compose([
            A.Resize(
                height=cls.IMG_SIZE,
                width=cls.IMG_SIZE,
                interpolation=1,  # Bilinear
                p=1.0
            ),
            A.Normalize(
                mean=cls.MEAN,
                std=cls.STD,
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(p=1.0),
        ], p=1.0)
    
    @classmethod
    def get_tta_transforms(cls):
        """Get test-time augmentation transforms."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        base_transform = A.Compose([
            A.Resize(
                height=cls.IMG_SIZE,
                width=cls.IMG_SIZE,
                interpolation=1,
                p=1.0
            ),
            A.Normalize(
                mean=cls.MEAN,
                std=cls.STD,
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(p=1.0),
        ])
        
        # Define TTA transforms
        tta_transforms = [
            base_transform,  # Original
            A.Compose([  # Horizontal flip
                A.HorizontalFlip(p=1.0),
                *base_transform.transforms
            ]),
            A.Compose([  # Vertical flip
                A.VerticalFlip(p=1.0),
                *base_transform.transforms
            ]),
            A.Compose([  # Rotate 90
                A.Rotate(limit=90, p=1.0),
                *base_transform.transforms
            ]),
            A.Compose([  # Rotate 180
                A.Rotate(limit=180, p=1.0),
                *base_transform.transforms
            ]),
        ]
        
        return tta_transforms[:cls.TTA_NUM_AUGS] if cls.TTA_NUM_AUGS > 1 else [base_transform]
        
    @classmethod
    def get_test_transforms(cls):
        """Get test data transformation pipeline with TTA support."""
        from albumentations import (
            Compose, Resize, Normalize, HorizontalFlip, VerticalFlip, RandomRotate90
        )
        from albumentations.pytorch import ToTensorV2
        
        if cls.TTA_NUM_AUGS <= 1:
            return cls.get_val_transforms()
            
        # Base transforms
        base_transforms = [
            Resize(height=cls.IMG_SIZE, width=cls.IMG_SIZE, p=1.0),
            Normalize(mean=cls.MEAN, std=cls.STD, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ]
        
        # Add TTA transforms
        tta_transforms = []
        for _ in range(cls.TTA_NUM_AUGS):
            tta_transforms.append(Compose([
                *base_transforms,
                RandomRotate90(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
            ], p=1.0))
            
        return tta_transforms
    
    @staticmethod
    def get_val_test_transforms() -> Compose:
        return Compose(
            [
                Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    
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
    def to_dict(cls) -> Dict[str, Union[str, int, float, bool]]:
        """Convert config to dictionary for logging."""
        config_dict = {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("_") and k.isupper() and not callable(getattr(cls, k))
        }
        # Add hardware info
        config_dict.update({"hardware_info": cls.get_hardware_info()})
        return config_dict
    
    @classmethod
    def log_hardware_info(cls):
        """Log hardware information."""
        hw_info = cls.get_hardware_info()
        print("\n" + "="*50)
        print("Hardware Configuration:")
        print("-"*50)
        for k, v in hw_info.items():
            print(f"{k.replace('_', ' ').title()}: {v}")
        print("="*50 + "\n")
    
    # Experiment tracking
    EXPERIMENT_NAME = "krishirakshak"
    LOG_EVERY_N_STEPS = 10
    
    # Reproducibility
    SEED = 42


    @classmethod
    def setup_training(cls, model: nn.Module, train_loader) -> tuple:
        """Setup training components."""
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

# Initialize directories
Config.setup_dirs()
