"""KrishiRakshak - Plant Disease Detection with Grad-CAM and Multilingual Support."""
import os
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image

# Add src to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.models.plant_model import PlantDiseaseModel
from src.utils.gradcam import apply_gradcam
from src.utils.translations import translator

# Constants
MODEL_PATH = "models/krishirakshak_model.pt"  # Update this path
example_images = [
    os.path.join("examples", f) for f in os.listdir("examples") 
    if f.endswith(('.jpg', '.jpeg', '.png'))
] if os.path.exists("examples") else []

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    """Load the trained model."""
    global model
    if model is None:
        config = Config()
        model = PlantDiseaseModel(config)
        
        # Load weights
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=device)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
            
            model = model.to(device)
            model.eval()
            print("Model loaded successfully")
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

def predict(image: np.ndarray, language: str = 'en') -> tuple:
    """
    Make prediction and generate Grad-CAM visualization.
    
    Args:
        image: Input image (numpy array)
        language: Language code ('en', 'hi', 'mr')
        
    Returns:
        tuple: (original_image, heatmap, overlay, prediction_text)
    """
    try:
        # Load model if not already loaded
        model = load_model()
        
        # Preprocess image
        input_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
        
        # Generate Grad-CAM
        heatmap, overlay = apply_gradcam(
            model=model,
            input_tensor=input_tensor,
            original_image=image,
            target_class=pred_class
        )
        
        # Get class name (replace with your actual class names)
        class_names = [f"Class_{i}" for i in range(38)]  # Update with actual class names
        pred_label = class_names[pred_class]
        
        # Translate prediction
        translated_label = translator.translate_disease(pred_label, language)
        
        # Prepare prediction text
        prediction_text = (
            f"{translator.translate('prediction', language)}: {translated_label}\n"
            f"{translator.translate('confidence', language)}: {confidence:.2f}"
        )
        
        return image, heatmap, overlay, prediction_text
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        error_msg = f"{translator.translate('error_processing', language)}: {str(e)}"
        return image, None, None, error_msg

def create_ui():
    """Create Gradio UI."""
    # Available languages
    languages = translator.get_available_languages()
    
    with gr.Blocks(title="KrishiRakshak - Plant Disease Detection") as demo:
        gr.Markdown("# 🌱 KrishiRakshak - Plant Disease Detection")
        gr.Markdown("Upload an image of a plant leaf to detect diseases.")
        
        with gr.Row():
            with gr.Column():
                # Language selector
                language = gr.Dropdown(
                    choices=list(languages.items()),
                    value='en',
                    label="Select Language"
                )
                
                # Image upload
                image_input = gr.Image(
                    label=translator.translate('upload_image'),
                    type="numpy"
                )
                
                # Buttons
                with gr.Row():
                    predict_btn = gr.Button(translator.translate('predict'))
                    clear_btn = gr.Button(translator.translate('clear'))
                
                # Add example images if available
                if example_images:
                    gr.Examples(
                        examples=example_images,
                        inputs=image_input,
                        label="Example Images"
                    )
            
            with gr.Column():
                # Output tabs
                with gr.Tabs():
                    with gr.TabItem("Original"):
                        original_output = gr.Image(label="Original Image")
                    
                    with gr.TabItem("Heatmap"):
                        heatmap_output = gr.Image(label="Grad-CAM Heatmap")
                    
                    with gr.TabItem("Overlay"):
                        overlay_output = gr.Image(label="Grad-CAM Overlay")
                
                # Prediction text
                text_output = gr.Textbox(
                    label=translator.translate('prediction'),
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
        def update_ui_text(lang):
            return {
                image_input: gr.update(label=translator.translate('upload_image', lang)),
                predict_btn: gr.update(value=translator.translate('predict', lang)),
                clear_btn: gr.update(value=translator.translate('clear', lang)),
                text_output: gr.update(label=translator.translate('prediction', lang)),
            }
        
        language.change(
            fn=update_ui_text,
            inputs=language,
            outputs=[image_input, predict_btn, clear_btn, text_output]
        )
    
    return demo

if __name__ == "__main__":
    # Create UI
    demo = create_ui()
    
    # Run with share=True for public link
    demo.launch(share=True)
