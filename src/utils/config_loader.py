"""
Configuration loader with backward compatibility.
Handles loading of the new unified config.yaml while maintaining
compatibility with the old configuration structure.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

class ConfigLoader:
    """Load and manage configuration with backward compatibility."""
    
    def __init__(self, config_path: str = None):
        """Initialize the config loader.
        
        Args:
            config_path: Path to the unified config file. If None, will look for config.yaml in the root.
        """
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
        self._warn_deprecated()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration file with validation."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set default values for backward compatibility
        self._set_defaults(config)
        
        return config
    
    def _set_defaults(self, config: Dict[str, Any]) -> None:
        """Set default values for backward compatibility."""
        # Set default project paths if not specified
        if 'project' not in config:
            config['project'] = {}
        
        project = config['project']
        project.setdefault('output_dir', 'output')
        project.setdefault('log_dir', 'logs')
        project.setdefault('checkpoint_dir', 'checkpoints')
        
        # Set default data paths
        if 'data' not in config:
            config['data'] = {}
        
        data = config['data']
        data.setdefault('csv_path', 'data/metadata.csv')
        data.setdefault('rgb_dir', 'data/images')
        data.setdefault('ms_dir', 'data/multispectral')
        data.setdefault('image_size', [224, 224])
    
    def _warn_deprecated(self) -> None:
        """Warn about deprecated config files that are no longer needed."""
        deprecated_files = [
            'configs/default.yaml',
            'configs/hybrid_model.yaml',
            'configs/train_config.yaml'
        ]
        
        for file in deprecated_files:
            if os.path.exists(file):
                warnings.warn(
                    f"Deprecated config file found: {file}. "
                    "This file is no longer needed and can be removed. "
                    "All configuration is now in config.yaml",
                    DeprecationWarning
                )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dict-style access."""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a dictionary."""
        return self.config
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the current configuration to a file."""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)


# Global config instance for easy import
config = ConfigLoader()

def get_config() -> ConfigLoader:
    """Get the global configuration instance."""
    return config
