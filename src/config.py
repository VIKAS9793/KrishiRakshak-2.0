"""Configuration for the KrishiRakshak ML pipeline."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import torch
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
    RandomResizedCrop,
    Rotate,
    VerticalFlip,
    Resize,
)
from albumentations.pytorch import ToTensorV2


class Config:
    # Detect hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hardware-specific settings
    if DEVICE == "cuda":
        # GPU settings
        BATCH_SIZE = 32
        NUM_WORKERS = 4
        PIN_MEMORY = True
        PREFETCH_FACTOR = 2
    else:
        # CPU settings (optimized for local hardware)
        BATCH_SIZE = 16  # Reduced batch size for CPU
        NUM_WORKERS = 2  # Reduced workers for CPU
        PIN_MEMORY = False
        PREFETCH_FACTOR = 1
    
    # Project structure
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
    FIXED_DATA_DIR = DATA_DIR / "fixed_data"
    MODEL_DIR = ROOT_DIR / "models"
    LOGS_DIR = ROOT_DIR / "logs"
    REPORTS_DIR = ROOT_DIR / "reports"
    SRC_DIR = ROOT_DIR / "src"
    
    # Create directories if they don't exist
    for dir_path in [MODEL_DIR, LOGS_DIR, REPORTS_DIR, FIXED_DATA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Model parameters
    MODEL_NAME = "efficientnet_b3"
    PRETRAINED = True
    NUM_CLASSES = 38  # Update this based on your actual number of classes
    DROPOUT_RATE = 0.2
    
    # Data parameters
    IMAGE_SIZE = (300, 300)  # EfficientNet-B3 default input size
    
    # Training parameters
    EPOCHS = 50
    LEARNING_RATE = 0.001 if DEVICE == "cuda" else 0.0005  # Lower LR for CPU
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    
    # Learning rate scheduler
    LR_SCHEDULER = "cosine"  # 'cosine' or 'reduce_on_plateau'
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 5
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    
    # Data augmentation
    @staticmethod
    def get_train_transforms() -> Compose:
        return Compose(
            [
                RandomResizedCrop(
                    height=Config.IMAGE_SIZE[0],
                    width=Config.IMAGE_SIZE[1],
                    scale=(0.7, 1.0),
                    ratio=(0.8, 1.2),
                ),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Rotate(limit=45, p=0.5),
                RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    
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


# Initialize directories
Config.setup_dirs()
