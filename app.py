"""
KrishiRakshak - Plant Disease Detection with Grad-CAM and Multilingual Support.

This is the main entry point for the Gradio web interface.
"""
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.models.plant_model import PlantDiseaseModel
from src.utils.gradcam import apply_gradcam
from src.utils.translations import translator

# Load example images
example_images = (
    [str(f) for f in EXAMPLE_IMAGES_DIR.glob("*.{jpg,jpeg,png}")]
    if EXAMPLE_IMAGES_DIR.exists()
    else []
)

# Constants
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "krishirakshak_model.pt"
EXAMPLE_IMAGES_DIR = Path("examples")

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model() -> PlantDiseaseModel:
    """
    Load the trained model from disk.
    
    Returns:
        PlantDiseaseModel: The loaded and configured model
    """
    global model
    if model is None:
        config = Config()
        model = PlantDiseaseModel(config)
        
        # Load weights if they exist
        if MODEL_PATH.exists():
            state_dict = torch.load(MODEL_PATH, map_location=device)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
            
            model = model.to(device)
            model.eval()
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return model

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess image for model input."""
    # Convert to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize and normalize
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor and add batch dimension
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    return image.unsqueeze(0).to(device)

def predict(
    image: np.ndarray, 
    language: str = 'en'
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    """
    Make prediction and generate Grad-CAM visualization.
    
    Args:
        image: Input image as a numpy array in RGB format
        language: Language code ('en', 'hi', 'mr')
        
    Returns:
        tuple: Contains:
            - Original image (numpy array)
            - Heatmap visualization (numpy array)
            - Overlay visualization (numpy array)
            - Prediction text (str)
    """
    try:
        # Convert Gradio's numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        # Load model if not already loaded
        model = load_model()
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = model.class_names[predicted_idx.item()]
        
        # Generate Grad-CAM
        heatmap, overlay = apply_gradcam(model, input_tensor, predicted_idx.item())
        
        # Get translated prediction
        prediction_text = translator.get_translation(
            f"prediction_{predicted_class}", 
            language
        )
        confidence_text = translator.get_translation("confidence", language)
        
        # Format output text
        result_text = f"{prediction_text}\n{confidence_text}: {confidence:.2%}"
        
        # Convert images to RGB for display
        original_img = np.array(image)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay_img = cv2.addWeighted(original_img, 0.7, heatmap_img, 0.3, 0)
        
        return original_img, heatmap_img, overlay_img, result_text
        
    except Exception as e:
        error_msg = translator.get_translation("prediction_error", language)
        error_text = f"{error_msg}: {str(e)}"
        print(error_text)
        return None, None, None, error_text

def create_ui() -> gr.Blocks:
    """
    Create and configure the Gradio web interface for KrishiRakshak.
    
    Returns:
        gr.Blocks: Configured Gradio Blocks interface
    """
    # Available languages
    languages = {
        'en': 'English',
        'hi': 'हिंदी',
        'mr': 'मराठी'
    }
    
    with gr.Blocks(title="KrishiRakshak - Plant Disease Detection") as demo:
        # Title and description
        gr.Markdown("# 🌱 KrishiRakshak - Plant Disease Detection")
        gr.Markdown("Upload an image of a plant leaf to detect diseases.")
        
        with gr.Row():
            with gr.Column():
                # Language selector
                language = gr.Dropdown(
                    choices=list(languages.values()),
                    value='English',
                    label="Select Language"
                )
                
                # Image upload
                image_input = gr.Image(
                    label=translator.get_translation("upload_image", "en"),
                    type="numpy"
                )
                
                # Buttons
                with gr.Row():
                    predict_btn = gr.Button(
                        translator.get_translation("predict", "en")
                    )
                    clear_btn = gr.Button(
                        translator.get_translation("clear", "en")
                    )
                
                # Example images
                if example_images:
                    gr.Examples(
                        examples=example_images,
                        inputs=image_input,
                        label=translator.get_translation("example_images", "en")
                    )
            
            with gr.Column():
                # Output tabs
                with gr.Tabs():
                    with gr.TabItem("Original"):
                        original_output = gr.Image(
                            label=translator.get_translation("original_image", "en")
                        )
                    
                    with gr.TabItem("Heatmap"):
                        heatmap_output = gr.Image(
                            label=translator.get_translation("heatmap", "en")
                        )
                    
                    with gr.TabItem("Overlay"):
                        overlay_output = gr.Image(
                            label=translator.get_translation("overlay", "en")
                        )
                
                # Prediction text
                text_output = gr.Textbox(
                    label=translator.get_translation("prediction", "en"),
                    interactive=False
                )
        
        # Event handlers
        predict_btn.click(
            fn=predict,
            inputs=[image_input, language],
            outputs=[original_output, heatmap_output, overlay_output, text_output]
        )
        
        clear_btn.click(
            lambda: [None] * 4,  # Clear all outputs
            inputs=[],
            outputs=[original_output, heatmap_output, overlay_output, text_output]
        )
        
        # Update UI text when language changes
        def update_ui_text(lang: str) -> dict:
            """Update UI text elements when language changes."""
            lang_code = {
                'English': 'en',
                'हिंदी': 'hi',
                'मराठी': 'mr'
            }.get(lang, 'en')
            
            return {
                image_input: gr.update(
                    label=translator.get_translation("upload_image", lang_code)
                ),
                predict_btn: gr.update(
                    value=translator.get_translation("predict", lang_code)
                ),
                clear_btn: gr.update(
                    value=translator.get_translation("clear", lang_code)
                ),
                text_output: gr.update(
                    label=translator.get_translation("prediction", lang_code)
                ),
            }
        
        language.change(
            fn=update_ui_text,
            inputs=language,
            outputs=[
                image_input, 
                predict_btn, 
                clear_btn, 
                text_output
            ]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the web interface
    app = create_ui()
    
    # Run with share=True for public link (set to False for local only)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True for public link
    )
