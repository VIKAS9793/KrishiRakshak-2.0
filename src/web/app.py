"""
Gradio UI for KrishiRakshak with full multilingual support.

This script creates a dynamic web interface where all text elements
(labels, buttons, titles, and results) are translated in real-time
using the Translator module.
"""
import gradio as gr
from PIL import Image
import numpy as np

# Assume the Translator class from the Canvas is saved in 'utils/translations.py'
# If the file is in the same directory, you can use: from your_file_name import translator
from utils.translations import translator

# --- Mock Model for Demonstration ---
class MockModel:
    """A mock model to simulate predictions."""
    def predict(self, image: Image.Image) -> dict:
        """
        Simulates a model prediction, returning a disease key and confidence.
        
        Args:
            image: A PIL Image (not used in mock, but required for interface).
            
        Returns:
            A dictionary containing the disease key, confidence, and a mock heatmap.
        """
        # Simulate a prediction result
        disease_key = 'Tomato___Early_blight'
        confidence = 0.92
        
        # Generate a random heatmap for visualization
        heatmap_array = np.random.rand(224, 224, 3) * 255
        heatmap_image = Image.fromarray(heatmap_array.astype('uint8'))
        
        return {
            'disease_key': disease_key,
            'confidence': confidence,
            'heatmap': heatmap_image,
        }

# --- UI Creation ---
def create_ui():
    """Builds and launches the Gradio web interface."""
    
    # Initialize the model
    model = MockModel()
    
    # Get available languages and create a mapping from name to code (e.g., "English" -> "en")
    available_langs = translator.get_available_languages()
    lang_name_to_code = {v: k for k, v in available_langs.items()}

    def get_predictions(image: Image.Image, lang_name: str) -> tuple:
        """
        Processes the image, gets predictions, and translates the results.
        
        Args:
            image: The user-uploaded image.
            lang_name: The selected language name from the UI (e.g., "English").
            
        Returns:
            A tuple containing the translated disease name, confidence score, and heatmap.
        """
        if image is None:
            # Get the language code to translate the error message
            lang_code = lang_name_to_code.get(lang_name, 'en')
            error_message = translator.translate_ui('no_image', lang_code)
            # Raise a Gradio error to display a popup to the user
            raise gr.Error(error_message)
        
        # Get the language code from the selected name
        lang_code = lang_name_to_code.get(lang_name, 'en')
        
        # Get model predictions
        result = model.predict(image)
        
        # Translate the disease name using the translator
        translated_disease = translator.translate_disease(result['disease_key'], lang_code)
        
        return translated_disease, f"{result['confidence']:.2%}", result['heatmap']

    def update_ui_language(lang_name: str) -> dict:
        """
        Updates all UI component labels and values when the language is changed.
        
        Args:
            lang_name: The new language name selected from the dropdown.
            
        Returns:
            A dictionary of gr.update() calls to dynamically change UI elements.
        """
        lang_code = lang_name_to_code.get(lang_name, 'en')
        
        return {
            app_title: gr.update(value=translator.translate_ui('app_title', lang_code)),
            image_input: gr.update(label=translator.translate_ui('upload_image', lang_code)),
            predict_btn: gr.update(value=translator.translate_ui('predict', lang_code)),
            clear_btn: gr.update(value=translator.translate_ui('clear', lang_code)),
            disease_label: gr.update(label=translator.translate_ui('prediction', lang_code)),
            confidence_label: gr.update(label=translator.translate_ui('confidence', lang_code)),
            heatmap_label: gr.update(label=translator.translate_ui('heatmap', lang_code)),
            language_dropdown: gr.update(label=translator.translate_ui('language', lang_code))
        }

    # --- Gradio Interface Definition ---
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # App Title
        app_title = gr.Markdown(f"# {translator.translate_ui('app_title', 'en')}")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Language Selector
                language_dropdown = gr.Dropdown(
                    choices=list(available_langs.values()),
                    value='English',
                    label="Language"
                )
                
                # Image Input
                image_input = gr.Image(
                    type="pil",
                    label=translator.translate_ui('upload_image', 'en'),
                    height=300
                )
                
                with gr.Row():
                    # Action Buttons
                    predict_btn = gr.Button(translator.translate_ui('predict', 'en'), variant="primary")
                    clear_btn = gr.Button(translator.translate_ui('clear', 'en'))
            
            with gr.Column(scale=1):
                # Output Components
                disease_label = gr.Label(label=translator.translate_ui('prediction', 'en'))
                confidence_label = gr.Label(label=translator.translate_ui('confidence', 'en'))
                heatmap_label = gr.Image(label=translator.translate_ui('heatmap', 'en'), height=300)

        # --- Event Listeners ---
        
        # 1. When the Predict button is clicked
        predict_btn.click(
            fn=get_predictions,
            inputs=[image_input, language_dropdown],
            outputs=[disease_label, confidence_label, heatmap_label]
        )
        
        # 2. When the Clear button is clicked
        clear_btn.click(
            fn=lambda: (None, None, None, None),
            outputs=[image_input, disease_label, confidence_label, heatmap_label]
        )
        
        # 3. When the language dropdown changes, update all UI text
        language_dropdown.change(
            fn=update_ui_language,
            inputs=language_dropdown,
            outputs=[
                app_title, image_input, predict_btn, clear_btn,
                disease_label, confidence_label, heatmap_label, language_dropdown
            ]
        )
        
    return demo

if __name__ == "__main__":
    # Create and launch the Gradio app
    krishirakshak_ui = create_ui()
    krishirakshak_ui.launch()
