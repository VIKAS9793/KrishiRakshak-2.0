"""
Production-Ready Grad-CAM Implementation for PyTorch Models.

This module provides a reusable GradCAM class designed to be compatible with
complex model architectures (like the HybridModel) and data formats.
It decouples the core Grad-CAM logic from visualization, enabling flexible use.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from typing import Dict, List, Optional, Tuple

class GradCAM:
    """
    A reusable class for generating Gradient-weighted Class Activation Maps (Grad-CAM).
    Designed to work with models expecting dictionary inputs and complex architectures.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to the target layer
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Saves the feature map activations from the forward pass."""
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        """Saves the gradients from the backward pass."""
        self.gradients = grad_output[0]

    def _get_cam_weights(self, grads: torch.Tensor) -> torch.Tensor:
        """Computes the alpha weights for CAM."""
        return torch.mean(grads, dim=(2, 3), keepdim=True)

    def generate_heatmap(
        self,
        input_batch: Dict[str, torch.Tensor],
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates the Grad-CAM heatmap.

        Args:
            input_batch (Dict[str, torch.Tensor]): The input batch dictionary, as expected
                by the model's forward pass.
            target_class (Optional[int]): The target class index. If None, the class with
                the highest score will be used.

        Returns:
            np.ndarray: The generated heatmap, normalized to [0, 1].
        """
        self.model.eval()
        
        # 1. Forward pass to get model output
        output = self.model(input_batch)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 2. Backward pass to get gradients
        self.model.zero_grad()
        # Use the score for the target class to compute gradients
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)

        # Ensure gradients and activations were captured
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations. Check hook registration.")

        # 3. Compute the heatmap
        pooled_gradients = self._get_cam_weights(self.gradients)
        # Get activations for the first image in the batch
        activations = self.activations[0].detach()
        
        # Weight the feature maps by the gradients
        heatmap = torch.sum(pooled_gradients[0] * activations, dim=0)
        heatmap = nn.functional.relu(heatmap)
        
        # Normalize the heatmap
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()

def get_target_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """
    Retrieves a nested layer from a model using its string name.
    
    Example: get_target_layer(model, 'rgb_backbone.conv_head')
    """
    current_module = model
    for part in layer_name.split('.'):
        current_module = getattr(current_module, part)
    if not isinstance(current_module, nn.Module):
        raise TypeError(f"The specified layer '{layer_name}' is not a valid nn.Module.")
    return current_module

def visualize_gradcam(
    heatmap: np.ndarray,
    image_pil: Image.Image,
    alpha: float = 0.6
) -> Image.Image:
    """
    Overlays a heatmap onto an image.

    Args:
        heatmap (np.ndarray): The normalized heatmap (H, W).
        image_pil (Image.Image): The original PIL image.
        alpha (float): The transparency of the heatmap overlay.

    Returns:
        Image.Image: The original image with the heatmap overlaid.
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (image_pil.width, image_pil.height))
    heatmap_resized = (heatmap_resized * 255).astype(np.uint8)

    # Apply a colormap to the heatmap
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Convert original image to numpy array
    image_np = np.array(image_pil)

    # Superimpose the heatmap on the original image
    overlaid_img_np = (heatmap_color * alpha + image_np * (1 - alpha)).astype(np.uint8)
    
    return Image.fromarray(overlaid_img_np)


# --- Example Usage with the KrishiSahayak Project ---
# This demonstrates how the new GradCAM class would be used.
def example_usage():
    from src.models.hybrid import HybridModel # Assuming this is our refactored HybridModel

    # 1. Load your trained model
    # model = HybridModel.load_from_checkpoint('path/to/your/checkpoint.ckpt')
    model = HybridModel(num_classes=38, backbone_name='efficientnet_b0') # Dummy model for demo

    # 2. Specify the target layer for visualization
    # This name must correspond to a layer in your HybridModel architecture.
    target_layer_name = 'rgb_backbone.conv_head'
    target_layer = get_target_layer(model, target_layer_name)

    # 3. Instantiate the GradCAM utility
    gradcam_generator = GradCAM(model=model, target_layer=target_layer)

    # 4. Prepare your input image and batch dictionary
    # This preprocessing should match what was used during training
    image_path = "path/to/your/rgb_image.jpg"
    image_pil = Image.open(image_path).convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image_pil).unsqueeze(0) # Add batch dimension

    # The batch must match the model's expected input format
    input_batch = {'image': input_tensor} 

    # 5. Generate the heatmap
    heatmap_np = gradcam_generator.generate_heatmap(input_batch)

    # 6. Visualize the result
    final_image = visualize_gradcam(heatmap_np, image_pil)
    final_image.save("gradcam_result.png")
    print("Grad-CAM image saved to gradcam_result.png")

if __name__ == '__main__':
    # This block can be used for testing the implementation
    # Note: You would need to replace the dummy model and image path with real ones.
    # example_usage()
    pass