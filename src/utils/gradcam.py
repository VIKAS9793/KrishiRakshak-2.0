import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class GradCAM:
    """
    Generates a Grad-CAM heatmap for a given model and target layer, indicating
    the regions of an image that are important for a specific class prediction.

    This implementation is designed to be robust and easy to use. It should be
    used as a context manager to ensure that hooks are properly removed.

    Key Improvements:
    - Robustness: Validates the target layer name to prevent crashes.
    - Clarity: Includes comprehensive docstrings and type hints.
    - Device-Agnostic: Automatically detects and uses the model's device (CPU/GPU).
    - Resource Management: Uses a context manager (`with`) for automatic hook cleanup.
    - Explicit Control: Allows specifying which image in a batch to process.

    Example Usage:
        # Assuming `model`, `input_tensor` are defined and on the correct device.
        try:
            # Use the last convolutional layer of a ResNet-like model
            with GradCAM(model=model, target_layer_name='layer4.2.conv3') as cam_generator:
                scores = model(input_tensor)
                class_idx = torch.argmax(scores[0]).item()
                heatmap = cam_generator(class_idx=class_idx, scores=scores)
        except (ValueError, RuntimeError) as e:
            print(f"Error: {e}")
    """

    def __init__(self, model: nn.Module, target_layer_name: str):
        """
        Args:
            model (nn.Module): The model to generate the CAM for.
            target_layer_name (str): The name of the target layer to hook into.
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.target_layer = self._find_target_layer(target_layer_name)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _find_target_layer(self, layer_name: str) -> nn.Module:
        """Finds the target layer module by its name and raises an error if not found."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        
        available_layers = [name for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d)]
        raise ValueError(
            f"Target layer '{layer_name}' not found. "
            f"Consider one of the following convolutional layers: {available_layers}"
        )

    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        """Saves the activations from the forward pass."""
        self.activations = output.detach().to(self.device)

    def _backward_hook(self, module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
        """Saves the gradients from the backward pass."""
        self.gradients = grad_output[0].detach().to(self.device)

    def _register_hooks(self):
        """Registers the forward and backward hooks."""
        self.hooks.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _remove_hooks(self):
        """Removes all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __call__(self, class_idx: int, scores: torch.Tensor, batch_idx: int = 0) -> np.ndarray:
        """
        Generates the Grad-CAM heatmap.

        Args:
            class_idx (int): The index of the class to generate the CAM for.
            scores (torch.Tensor): The output logits/scores from the model.
            batch_idx (int, optional): The index of the image in the batch to process. Defaults to 0.

        Returns:
            np.ndarray: A 2D numpy array representing the normalized heatmap (0 to 1).
        """
        if not self.hooks:
            raise RuntimeError("Hooks are not registered. Use as a context manager ('with' statement).")
            
        one_hot = torch.zeros_like(scores, device=self.device)
        one_hot[batch_idx, class_idx] = 1
        
        self.model.zero_grad()
        scores.backward(gradient=one_hot, retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Could not retrieve activations or gradients.")

        activations = self.activations[batch_idx]
        gradients = self.gradients[batch_idx]
        weights = F.adaptive_avg_pool2d(gradients, 1)
        
        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)

        cam -= torch.min(cam)
        cam /= (torch.max(cam) + 1e-8)

        return cam.cpu().numpy()

    def __enter__(self):
        self._register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_hooks()


def apply_gradcam_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Applies a Grad-CAM heatmap overlay on the original image.
    
    Args:
        original_image: Input image in RGB format, shape (H, W, 3), values in [0, 255]
        heatmap: 2D heatmap, shape (H', W'), values in [0, 1]
        alpha: Opacity of the heatmap overlay [0, 1]
        colormap: OpenCV colormap constant (e.g., cv2.COLORMAP_JET)
        
    Returns:
        Overlayed image in RGB format, same shape as original_image
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        
    if not (isinstance(original_image, np.ndarray) and original_image.ndim == 3):
        raise ValueError(f"original_image must be 3D numpy array, got {type(original_image)}")
        
    if not (isinstance(heatmap, np.ndarray) and heatmap.ndim == 2):
        raise ValueError(f"heatmap must be 2D numpy array, got {type(heatmap)}")
        
    if not (0 <= heatmap.min() and heatmap.max() <= 1.0001): # Allow for small floating point inaccuracies
        raise ValueError(f"heatmap values must be in range [0, 1], but found min: {heatmap.min()}, max: {heatmap.max()}")
    
    # Convert heatmap to uint8 and apply colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Resize heatmap to match original image dimensions
    if heatmap.shape != original_image.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, 
                                     (original_image.shape[1], original_image.shape[0]))
    
    # Convert to BGR for OpenCV if it's a color image
    if original_image.shape[2] == 3: # RGB to BGR
        original_bgr = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else: # Grayscale
        original_bgr = original_image.astype(np.uint8)
    
    # Blend images
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Convert back to RGB for consistency
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)