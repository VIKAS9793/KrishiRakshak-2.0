"""
Configuration utilities for the KrishiSahayak project.

This module provides functions for loading, validating, and saving configuration files.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: Union[str, Path], resolve_paths: bool = True) -> Dict[str, Any]:
    """
    Load a YAML configuration file and optionally resolve relative paths.
    
    Args:
        config_path: Path to the YAML configuration file.
        resolve_paths: If True, resolve all paths relative to the config file.
        
    Returns:
        Dictionary containing the configuration.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise
    
    # Set the config directory as the base for relative paths
    config_dir = config_path.parent
    
    def _resolve_paths(config_dict: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        """Recursively resolve paths in the config dictionary."""
        resolved = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                resolved[key] = _resolve_paths(value, base_dir)
            elif isinstance(value, str) and 'path' in key.lower() and value:
                # Only resolve paths that contain 'path' in the key
                try:
                    # Skip URLs and environment variables
                    if not (value.startswith(('http://', 'https://', 's3://', '$'))):
                        resolved_path = (base_dir / value).resolve()
                        resolved[key] = str(resolved_path)
                    else:
                        resolved[key] = value
                except (TypeError, ValueError):
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved
    
    if resolve_paths:
        config = _resolve_paths(config, config_dir)
    
    # Add the config directory and file path to the config
    config['_meta'] = {
        'config_dir': str(config_dir),
        'config_file': str(config_path),
        'cwd': str(Path.cwd())
    }
    
    return config

def save_config(config: Dict[str, Any], output_path: Union[str, Path], format: str = 'yaml') -> None:
    """
    Save a configuration dictionary to a file.
    
    Args:
        config: Configuration dictionary to save.
        output_path: Path where the configuration will be saved.
        format: Output format, either 'yaml' or 'json'.
        
    Raises:
        ValueError: If an unsupported format is provided.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format.lower() == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2, sort_keys=True)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'.")
        
        logger.info(f"Configuration saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving configuration to {output_path}: {e}")
        raise

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a configuration dictionary with new values.
    
    Args:
        config: The original configuration dictionary.
        updates: Dictionary containing updates to apply.
        
    Returns:
        Updated configuration dictionary.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    return config

def validate_config(config: Dict[str, Any], required_keys: Optional[list] = None) -> bool:
    """
    Validate that a configuration contains all required keys.
    
    Args:
        config: Configuration dictionary to validate.
        required_keys: List of required top-level keys. If None, uses a default set.
        
    Returns:
        bool: True if the configuration is valid.
        
    Raises:
        ValueError: If required keys are missing from the configuration.
    """
    if required_keys is None:
        required_keys = [
            'data', 'model', 'training', 'optimizer', 'scheduler',
            'callbacks', 'logging', 'seed'
        ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    # Validate data paths exist if they are specified
    if 'data' in config:
        data_cfg = config['data']
        paths_to_check = [
            data_cfg.get('train_data_dir'),
            data_cfg.get('val_data_dir'),
            data_cfg.get('test_data_dir'),
            data_cfg.get('processed_dir')
        ]
        
        for path in paths_to_check:
            if path and not Path(path).exists():
                logger.warning(f"Warning: Path does not exist: {path}")
    
    return True
