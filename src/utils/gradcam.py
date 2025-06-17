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
    It is designed to be used as a context manager to ensure hooks are properly removed.
    
    Attributes:
        model: The PyTorch model to analyze.
        target_layer: The target convolutional layer module.
        gradient: Stores gradients from the backward pass.
        activation: Stores activations from the forward pass.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        """
        Initialize Grad-CAM with a model and target layer name.
        
        Args:
            model: PyTorch model to analyze.
            target_layer_name: Name of the target convolutional layer (e.g., 'features.18').
            
        Raises:
            ValueError: If the target layer is not found in the model.
        """
        self.model = model.eval() # Set model to evaluation mode
        self.target_layer = self._find_target_layer(target_layer_name)
        if self.target_layer is None:
            raise ValueError(f"Target layer '{target_layer_name}' not found in model.")

        self.gradient: Optional[torch.Tensor] = None
        self.activation: Optional[torch.Tensor] = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    def _find_target_layer(self, layer_name: str) -> Optional[torch.nn.Module]:
        """Utility to find a module by its string name."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        return None

    def _capture_gradient(self, grad: torch.Tensor) -> None:
        """Hook for capturing the gradient of the target layer's output."""
        self.gradient = grad

    def _capture_activation(self, module: torch.nn.Module, input_tensors: Tuple[torch.Tensor], output_tensor: torch.Tensor) -> None:
        """Hook for capturing the activation and registering a gradient hook."""
        self.activation = output_tensor.detach()
        # Register a backward hook on the activation tensor to capture gradients
        self.hook_handles.append(output_tensor.register_hook(self._capture_gradient))

    def _normalize_cam(self, cam: np.ndarray) -> np.ndarray:
        """Normalize the class activation map to the [0, 1] range."""
        cam = np.maximum(cam, 0)
        min_val, max_val = np.min(cam), np.max(cam)
        
        # Avoid division by zero
        if max_val - min_val > 1e-7:
            cam = (cam - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(cam)
            
        return cam

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate the Grad-CAM heatmap.
        
        This method orchestrates the forward and backward passes to compute the heatmap.
        
        Args:
            input_tensor: Input image tensor of shape (1, C, H, W).
            target_class: Target class index. If None, the predicted class is used.
            
        Returns:
            Normalized class activation map as a numpy array (H, W).
            
        Raises:
            RuntimeError: If hooks fail to capture activation or gradient.
        """
        # Register the forward hook
        forward_hook = self.target_layer.register_forward_hook(self._capture_activation)
        self.hook_handles.append(forward_hook)

        # 1. Forward pass
        with torch.set_grad_enabled(True):
            output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 2. Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=False)
        
        # Ensure hooks have captured the necessary data
        if self.activation is None or self.gradient is None:
            self.remove_hooks()
            raise RuntimeError("Failed to capture activation or gradient. Check target layer and model structure.")

        # 3. Compute CAM
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activation, dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        
        # Clean up all hooks immediately after use
        self.remove_hooks()
        
        return self._normalize_cam(cam)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


def apply_gradcam(
    grad_cam_instance: GradCAM,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_class: Optional[int] = None,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Grad-CAM to an image and generate a visual overlay.
    
    Args:
        grad_cam_instance: An initialized GradCAM object.
        input_tensor: Preprocessed input tensor (1, C, H, W).
        original_image: Original image (H, W, C) in RGB format with values in [0, 255].
        target_class: Target class index. If None, uses the predicted class.
        alpha: Opacity of the heatmap overlay [0, 1].
        
    Returns:
        A tuple of (heatmap, overlay_image) as numpy arrays. The overlay is in RGB format.
    """
    if not (isinstance(original_image, np.ndarray) and original_image.ndim == 3):
        raise ValueError("original_image must be a 3D numpy array (H, W, C).")
    
    # Generate heatmap data (normalized to [0, 1])
    heatmap_data = grad_cam_instance.generate(input_tensor, target_class)
    
    # Create a colored heatmap
    heatmap = (heatmap_data * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Resize heatmap to match the original image
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Superimpose the heatmap on the original image
    # Convert original image to BGR for OpenCV
    image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)
    
    # Convert the final overlay back to RGB
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return heatmap, overlay_rgb