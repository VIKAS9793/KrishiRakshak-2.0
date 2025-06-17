"""
Gradient-weighted Class Activation Mapping (Grad-CAM) implementation for PyTorch models.

This module provides functionality to generate Grad-CAM visualizations for CNN models,
highlighting the regions of the input image that were most important for the model's prediction.
"""
from typing import Tuple, Optional, List, Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
import cv2


class GradCAM:
    """
    Compute Gradient-weighted Class Activation Mapping (Grad-CAM) for PyTorch models.
    
    This class uses forward and backward hooks to capture activations and gradients
    from a target convolutional layer to generate class-discriminative localization maps.
    
    Attributes:
        model: The PyTorch model to analyze.
        target_layer: Name of the target convolutional layer.
        gradient: Stores gradients from the backward pass.
        activation: Stores activations from the forward pass.
        hook_handles: List to keep track of registered hooks for cleanup.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str):
        """
        Initialize Grad-CAM with a model and target layer.
        
        Args:
            model: PyTorch model to analyze.
            target_layer: Name of the target convolutional layer (e.g., 'features.18').
            
        Raises:
            ValueError: If the target layer is not found in the model.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradient: Optional[torch.Tensor] = None
        self.activation: Optional[torch.Tensor] = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()
    
    def _store_gradient(self, grad_output: torch.Tensor) -> None:
        """Store gradients from the backward pass."""
        self.gradient = grad_output[0] if grad_output[0] is not None else None
    
    def _store_activation_and_register_gradient_hook(
        self, 
        module: torch.nn.Module, 
        input: Tuple[torch.Tensor], 
        output: torch.Tensor
    ) -> None:
        """Store activations and register gradient hook for the current forward pass."""
        self.activation = output.detach()
        # Register backward hook on the output tensor
        self.hook_handles.append(
            output.register_hook(self._store_gradient)
        )
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        found_layer = False
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook_handles.append(
                    module.register_forward_hook(self._store_activation_and_register_gradient_hook)
                )
                found_layer = True
                break
                
        if not found_layer:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model.")
    
    def _normalize_cam(self, cam: np.ndarray) -> np.ndarray:
        """
        Normalize the class activation map to [0, 1] range.
        
        Args:
            cam: Raw class activation map.
            
        Returns:
            Normalized activation map.
        """
        cam = np.maximum(cam, 0)
        min_val = np.min(cam)
        max_val = np.max(cam)
        
        # Handle case where all values are the same
        if max_val == min_val:
            return np.zeros_like(cam)
            
        cam = (cam - min_val) / (max_val - min_val + 1e-7)
        return cam
    
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given input and target class.
        
        Args:
            input_tensor: Input image tensor of shape (1, C, H, W).
            target_class: Target class index. If None, uses the predicted class.
            
        Returns:
            Normalized class activation map as a numpy array (H, W).
            
        Raises:
            RuntimeError: If no gradient or activation was captured.
        """
        if self.activation is None or self.gradient is None:
            raise RuntimeError("No activation or gradient captured. Ensure forward pass completed successfully.")
            
        # Forward pass
        self.model.eval()
        with torch.set_grad_enabled(True):
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Backward pass for target class
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0][target_class] = 1.0
            output.backward(gradient=one_hot, retain_graph=False)
        
        # Compute CAM
        if self.gradient is None or self.activation is None:
            raise RuntimeError("Gradient or activation not captured during backward pass.")
            
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activation, dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().detach().cpu().numpy()
        
        return self._normalize_cam(cam)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks to prevent memory leaks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures hooks are removed."""
        self.remove_hooks()


def apply_gradcam(
    grad_cam: GradCAM,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_class: Optional[int] = None,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Grad-CAM to an input image and generate visualization.
    
    Args:
        grad_cam: Initialized GradCAM instance.
        input_tensor: Preprocessed input tensor (1, C, H, W).
        original_image: Original image (H, W, C) in RGB format with values in [0, 255].
        target_class: Target class index. If None, uses predicted class.
        alpha: Opacity of heatmap overlay [0, 1].
        
    Returns:
        Tuple of (heatmap, overlay_image) as numpy arrays.
        
    Raises:
        ValueError: If input validation fails.
    """
    if not isinstance(original_image, np.ndarray) or len(original_image.shape) != 3:
        raise ValueError("original_image must be a 3D numpy array (H, W, C)")
        
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1")
    
    # Generate heatmap data (normalized to [0, 1])
    heatmap_data = grad_cam.generate(input_tensor, target_class)
    
    # Convert heatmap to color (0-255)
    heatmap = (heatmap_data * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Resize heatmap to match original image dimensions
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Ensure original image is in BGR format and uint8
    if original_image.dtype == np.float32 and np.max(original_image) <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    
    if original_image.shape[2] == 3:  # RGB to BGR if needed
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
    # Create overlay
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    
    return heatmap, overlay
