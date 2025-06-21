"""
Weights & Biases (wandb) integration utilities.
"""
import os
from typing import Dict, Optional, Any

import wandb
from omegaconf import DictConfig

class WandbLogger:
    """Weights & Biases logger wrapper for experiment tracking."""
    
    def __init__(self, config: DictConfig, **kwargs):
        """
        Initialize wandb logger.
        
        Args:
            config: Hydra config object
            **kwargs: Additional arguments to pass to wandb.init
        """
        self.config = config
        self.run = None
        self._init_wandb(**kwargs)
    
    def _init_wandb(self, **kwargs):
        """Initialize wandb run with project configuration."""
        wandb_config = {
            "project": "KrishiSahayak",
            "entity": "vikassahani17-aditya-birla-capital",
            "group": self.config.get("experiment", {}).get("group", "default"),
            "tags": self.config.get("experiment", {}).get("tags", []),
            "config": self._convert_config(self.config),
            **kwargs
        }
        
        # Initialize wandb run
        self.run = wandb.init(**wandb_config)
        
        # Log code
        if self.config.get("experiment", {}).get("watch_model", False):
            self.watch_model()
    
    @staticmethod
    def _convert_config(config: DictConfig) -> Dict[str, Any]:
        """Convert OmegaConf config to Python dict for wandb."""
        # Convert to dict and handle any OmegaConf specific types
        config_dict = dict(config)
        
        # Handle nested configs
        for k, v in config_dict.items():
            if hasattr(v, "__dict__"):
                config_dict[k] = dict(v)
        
        return config_dict
    
    def watch_model(self, model=None, log: str = "gradients", log_freq: int = 1000):
        """Watch model for logging gradients and parameters."""
        if model is None:
            model = self.config.model
        
        wandb.watch(
            model,
            log=log,
            log_freq=log_freq,
            log_graph=True
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.run is not None:
            wandb.log(metrics, step=step)
    
    def log_artifact(self, file_path: str, name: str, type: str = "model"):
        """Log an artifact (model, dataset, etc.) to wandb."""
        if self.run is not None:
            artifact = wandb.Artifact(name, type=type)
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)
    
    def finish(self):
        """Finish the wandb run."""
        if self.run is not None:
            wandb.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        

def init_wandb(config: DictConfig, **kwargs) -> WandbLogger:
    """Initialize wandb logger with the given config.
    
    Example:
        ```python
        with init_wandb(cfg) as logger:
            # Your training loop
            logger.log_metrics({"loss": 0.1, "accuracy": 0.9})
        ```
    """
    return WandbLogger(config, **kwargs)
