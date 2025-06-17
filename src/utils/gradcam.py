"""Grad-CAM implementation for model explainability."""
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Tuple, Optional

class GradCAM:
    """Grad-CAM implementation for PyTorch models."""
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target convolutional layer to compute Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        
        # Register hooks
        self.hook_handles = []
        self._register_hooks()
    
    def _get_activation_hook(self, grad_output):
        """Hook to store gradients during backward pass."""
        self.gradient = grad_output[0]
    
    def _get_gradient_hook(self, module, input, output):
        """Hook to store activations during forward pass."""
        self.activation = output
        self.hook_handles.append(
            output.register_hook(self._get_activation_hook)
        )
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(self._get_gradient_hook)
                break
    
    def _normalize_cam(self, cam):
        """Normalize CAM to 0-1 range."""
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        return cam
    
    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index. If None, uses predicted class.
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute CAM
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activation, dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().detach().cpu().numpy()
        
        # Normalize and return
        return self._normalize_cam(cam)

def apply_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_layer: str = 'features.18',  # Last conv layer in MobileNetV2
    target_class: Optional[int] = None,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Grad-CAM to an input image.
    
    Args:
        model: PyTorch model
        input_tensor: Preprocessed input tensor (1, C, H, W)
        original_image: Original image (H, W, C) in RGB format
        target_layer: Name of target layer
        target_class: Target class index. If None, uses predicted class.
        alpha: Opacity of heatmap overlay
        
    Returns:
        Tuple of (heatmap, overlay_image)
    """
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap = grad_cam.generate(input_tensor, target_class)
    
    # Convert to uint8
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Convert original image to BGR if needed
    if original_image.shape[2] == 3 and len(original_image.shape) == 3:
        if original_image.dtype == np.float32 and np.max(original_image) <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    
    return heatmap, overlay
