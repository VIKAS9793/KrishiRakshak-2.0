"""
Utilities for setting random seeds for reproducibility.

This module provides functions to set random seeds for various libraries
(PyTorch, NumPy, Python's random, etc.) to ensure reproducible results.
"""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: If True, sets the CuDNN backend to deterministic mode.
                      This can impact performance but ensures reproducibility.
    """
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set CUDA random seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # Set CuDNN to deterministic mode if specified (may impact performance)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False  # Disable CuDNN to maximize reproducibility
            
            # Additional environment variables for deterministic behavior
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['PYTHONHASHSEED'] = str(seed)
            
    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")

def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for PyTorch DataLoader.
    
    This ensures that each worker process has a different, but deterministic seed.
    
    Args:
        worker_id: ID of the worker process.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class SeedContext:
    """Context manager for setting and restoring random seeds."""
    
    def __init__(self, seed: int, deterministic: bool = True):
        """
        Initialize the context manager.
        
        Args:
            seed: Random seed value.
            deterministic: If True, sets the CuDNN backend to deterministic mode.
        """
        self.seed = seed
        self.deterministic = deterministic
        self.previous_state = {}
    
    def __enter__(self):
        """Save the current random state and set the new seed."""
        # Save previous state
        self.previous_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'cudnn': {
                'deterministic': torch.backends.cudnn.deterministic,
                'benchmark': torch.backends.cudnn.benchmark,
                'enabled': torch.backends.cudnn.enabled
            } if torch.cuda.is_available() else None
        }
        
        # Set new seed
        set_seed(self.seed, self.deterministic)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the previous random state."""
        # Restore previous state
        random.setstate(self.previous_state['python'])
        np.random.set_state(self.previous_state['numpy'])
        torch.set_rng_state(self.previous_state['torch'].cpu())
        
        if torch.cuda.is_available() and self.previous_state['cuda'] is not None:
            torch.cuda.set_rng_state_all(self.previous_state['cuda'])
            
            if self.previous_state['cudnn'] is not None:
                torch.backends.cudnn.deterministic = self.previous_state['cudnn']['deterministic']
                torch.backends.cudnn.benchmark = self.previous_state['cudnn']['benchmark']
                torch.backends.cudnn.enabled = self.previous_state['cudnn']['enabled']

def get_random_state() -> dict:
    """
    Get the current random state of all relevant libraries.
    
    Returns:
        Dictionary containing the random states.
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'cudnn': {
            'deterministic': torch.backends.cudnn.deterministic,
            'benchmark': torch.backends.cudnn.benchmark,
            'enabled': torch.backends.cudnn.enabled
        } if torch.cuda.is_available() else None
    }
    return state

def set_random_state(state: dict) -> None:
    """
    Set the random state from a previously saved state.
    
    Args:
        state: Dictionary containing random states (as returned by get_random_state).
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    
    if torch.cuda.is_available() and state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
        
        if state['cudnn'] is not None:
            torch.backends.cudnn.deterministic = state['cudnn']['deterministic']
            torch.backends.cudnn.benchmark = state['cudnn']['benchmark']
            torch.backends.cudnn.enabled = state['cudnn']['enabled']

# Example usage:
if __name__ == "__main__":
    # Set a global seed
    set_seed(42)
    
    # Use a context manager for a specific block of code
    with SeedContext(42):
        # Code here will use seed=42
        pass
    
    # After the context manager, the previous random state is restored
