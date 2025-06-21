"""
Logging configuration and utilities for the KrishiSahayak project.

This module provides a centralized way to configure logging across the project,
including file and console handlers, formatting, and log level management.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
import yaml
import datetime

# Global logger instance
logger = None

def setup_logging(
    config: Optional[Union[Dict[str, Any], str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    log_level: Union[int, str] = logging.INFO,
    console: bool = True,
    file_logging: bool = True,
    log_format: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    error_log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        config: Configuration dictionary or path to config file (YAML/JSON).
        log_dir: Directory to save log files. If None, uses 'logs' in the current directory.
        log_level: Default log level (can be overridden by config).
        console: Whether to log to console.
        file_logging: Whether to log to files.
        log_format: Log format string. If None, uses a default format.
        log_file: Path to the main log file. If None, generates a timestamped filename.
        error_log_file: Path to the error log file. If None, generates a timestamped filename.
        
    Returns:
        Configured logger instance.
    """
    global logger
    
    # If logger is already configured, return it
    if logger is not None and len(logger.handlers) > 0:
        return logger
    
    # Parse config if provided as a file path
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if config_path.suffix.lower() in ('.yaml', '.yml'):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
    
    # Get logging configuration from config or use defaults
    if config and isinstance(config, dict):
        log_config = config.get('logging', {})
        log_dir = log_config.get('log_dir', log_dir)
        log_level = log_config.get('log_level', log_level)
        console = log_config.get('console', console)
        file_logging = log_config.get('file_logging', file_logging)
        log_format = log_config.get('log_format', log_format)
        log_file = log_config.get('log_file', log_file)
        error_log_file = log_config.get('error_log_file', error_log_file)
    
    # Convert log_level from string if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Set default log directory if not specified
    if log_dir is None:
        log_dir = Path('logs')
    else:
        log_dir = Path(log_dir)
    
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default log format if not specified
    if log_format is None:
        log_format = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s '
            '(%(filename)s:%(lineno)d)'
        )
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler if enabled
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handlers if enabled
    if file_logging:
        # Generate timestamp for log filenames
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Main log file
        if log_file is None:
            log_file = log_dir / f'krishisahayak_{timestamp}.log'
        else:
            log_file = Path(log_file)
        
        # Error log file
        if error_log_file is None:
            error_log_file = log_dir / f'krishisahayak_error_{timestamp}.log'
        else:
            error_log_file = Path(error_log_file)
        
        # Create file handlers
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        error_file_handler = logging.FileHandler(error_log_file)
        error_file_handler.setFormatter(formatter)
        error_file_handler.setLevel(logging.ERROR)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(error_file_handler)
        
        logger.info(f"Logging to file: {log_file}")
        logger.info(f"Error logging to file: {error_log_file}")
    
    # Set up root logger
    logging.basicConfig(level=log_level, handlers=logger.handlers)
    
    # Configure third-party loggers
    for lib in ['matplotlib', 'PIL', 'urllib3', 'tensorboard']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    # Log configuration
    logger.info("Logging configured successfully")
    logger.debug(f"Log level set to: {logging.getLevelName(log_level)}")
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name of the logger. If None, returns the root logger.
        
    Returns:
        Configured logger instance.
    """
    if name is None:
        return logging.getLogger()
    return logging.getLogger(name)

def log_config(config: dict, logger: logging.Logger = None):
    """
    Log the configuration dictionary in a readable format.
    
    Args:
        config: Configuration dictionary to log.
        logger: Logger instance to use. If None, uses the root logger.
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("Configuration:")
    for section, values in config.items():
        logger.info(f"  {section}:")
        if isinstance(values, dict):
            for key, value in values.items():
                logger.info(f"    {key}: {value}")
        else:
            logger.info(f"    {values}")

# Initialize the default logger when the module is imported
setup_logging()
