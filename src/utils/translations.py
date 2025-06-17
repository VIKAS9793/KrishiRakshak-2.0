"""
Translation module for KrishiRakshak.

This module centralizes all text translations for the application,
including UI elements and disease names, to support multilingual functionality.
"""
from typing import Dict, Optional

# ==============================================================================
# TRANSLATION DICTIONARIES
# For scalability, consider moving these dictionaries into separate JSON files
# (e.g., /translations/en.json, /translations/hi.json).
# ==============================================================================

# --- UI Text Translations ---
UI_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    'en': {
        'app_title': 'KrishiRakshak - Plant Disease Detection',
        'language_name': 'English',
        'upload_image': 'Upload an image of a plant leaf',
        'predict': 'Predict',
        'clear': 'Clear',
        'prediction': 'Prediction',
        'confidence': 'Confidence',
        'heatmap': 'Heatmap',
        'language': 'Language',
        'loading': 'Loading...',
        'error': 'Error',
        'no_image': 'Please upload an image first.',
    },
    'hi': {
        'app_title': 'कृषि रक्षक - पौध रोग पहचान',
        'language_name': 'हिंदी',
        'upload_image': 'पौधे की पत्ती की एक तस्वीर अपलोड करें',
        'predict': 'भविष्यवाणी करें',
        'clear': 'साफ़ करें',
        'prediction': 'भविष्यवाणी',
        'confidence': 'विश्वास स्तर',
        'heatmap': 'गर्मी का नक्शा',
        'language': 'भाषा',
        'loading': 'लोड हो रहा है...',
        'error': 'त्रुटि',
        'no_image': 'कृपया पहले एक तस्वीर अपलोड करें।',
    },
    'mr': {
        'app_title': 'कृषीरक्षक - वनस्पती रोग ओळख',
        'language_name': 'मराठी',
        'upload_image': 'वनस्पतीच्या पानाची प्रतिमा अपलोड करा',
        'predict': 'अंदाज लावा',
        'clear': 'साफ करा',
        'prediction': 'अंदाज',
        'confidence': 'आत्मविश्वास',
        'heatmap': 'हीटमॅप',
        'language': 'भाषा',
        'loading': 'लोड होत आहे...',
        'error': 'त्रुटी',
        'no_image': 'कृपया आधी एक प्रतिमा अपलोड करा.',
    }
}

# --- Disease Name Translations ---
DISEASE_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # Apple
    'Apple___Apple_scab': {'en': 'Apple Scab', 'hi': 'सेब का खरोंच', 'mr': 'सफरचंद खरुज'},
    'Apple___Black_rot': {'en': 'Apple Black Rot', 'hi': 'सेब का काला सड़न', 'mr': 'सफरचंद काळा कुजवा'},
    'Apple___Cedar_apple_rust': {'en': 'Cedar Apple Rust', 'hi': 'देवदार सेब की जंग', 'mr': 'देवदार सफरचंद गंज'},
    
    # Cherry
    'Cherry_(including_sour)___Powdery_mildew': {'en': 'Cherry Powdery Mildew', 'hi': 'चेरी का पाउडर फफूंद', 'mr': 'चेरी पावडरी मिल्ड्यू'},
    
    # Corn (Maize)
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'en': 'Corn Gray Leaf Spot', 'hi': 'मक्के का ग्रे पत्ती धब्बा', 'mr': 'मका करडा पान ठिपके'},
    'Corn_(maize)___Common_rust_': {'en': 'Corn Common Rust', 'hi': 'मक्के का सामान्य जंग', 'mr': 'मक्याचा सामान्य गंज'},
    'Corn_(maize)___healthy': {'en': 'Healthy Corn Plant', 'hi': 'स्वस्थ मक्के का पौधा', 'mr': 'निरोगी मक्याचे झाड'},

    # Grape
    'Grape___Black_rot': {'en': 'Grape Black Rot', 'hi': 'अंगूर का काला सड़न', 'mr': 'द्राक्ष काळा कुजवा'},
    
    # Potato
    'Potato___Early_blight': {'en': 'Potato Early Blight', 'hi': 'आलू का अगेती झुलसा', 'mr': 'बटाटा लवकर करपा'},
    'Potato___Late_blight': {'en': 'Potato Late Blight', 'hi': 'आलू का पछेती झुलसा', 'mr': 'बटाटा उशिरा करपा'},
    'Potato___healthy': {'en': 'Healthy Potato Plant', 'hi': 'स्वस्थ आलू का पौधा', 'mr': 'निरोगी बटाट्याचे झाड'},
    
    # Tomato
    'Tomato___Bacterial_spot': {'en': 'Tomato Bacterial Spot', 'hi': 'टमाटर का जीवाणु धब्बा', 'mr': 'टोमॅटो जिवाणू ठिपके'},
    'Tomato___Early_blight': {'en': 'Tomato Early Blight', 'hi': 'टमाटर का अगेती झुलसा', 'mr': 'टोमॅटो लवकर करपा'},
    'Tomato___Late_blight': {'en': 'Tomato Late Blight', 'hi': 'टमाटर का पछेती झुलसा', 'mr': 'टोमॅटो उशिरा करपा'},
    'Tomato___Leaf_Mold': {'en': 'Tomato Leaf Mold', 'hi': 'टमाटर की पत्ती का फफूंद', 'mr': 'टोमॅटो पानांवरची बुरशी'},
    'Tomato___Septoria_leaf_spot': {'en': 'Tomato Septoria Leaf Spot', 'hi': 'टमाटर का सेप्टोरिया पत्ती धब्बा', 'mr': 'टोमॅटो सेप्टोरिया पान ठिपके'},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'en': 'Tomato Spider Mites', 'hi': 'टमाटर का मकड़ी घुन', 'mr': 'टोमॅटो कोळी कीटक'},
    'Tomato___Target_Spot': {'en': 'Tomato Target Spot', 'hi': 'टमाटर का टारगेट स्पॉट', 'mr': 'टोमॅटो टार्गेट स्पॉट'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'en': 'Tomato Yellow Leaf Curl Virus', 'hi': 'टमाटर का पीला पत्ती मरोड़ वायरस', 'mr': 'टोमॅटो पिवळा पानावरील विषाणू'},
    'Tomato___Tomato_mosaic_virus': {'en': 'Tomato Mosaic Virus', 'hi': 'टमाटर मोज़ेक वायरस', 'mr': 'टोमॅटो मोझॅक विषाणू'},
    'Tomato___healthy': {'en': 'Healthy Tomato Plant', 'hi': 'स्वस्थ टमाटर का पौधा', 'mr': 'निरोगी टोमॅटोचे झाड'},
}


# ==============================================================================
# TRANSLATOR CLASS
# ==============================================================================

class Translator:
    """A simple translation service for providing multilingual support."""
    
    def __init__(self, default_lang: str = 'en'):
        """
        Initializes the translator.
        
        Args:
            default_lang: The default language code (e.g., 'en', 'hi').
        """
        self.default_lang = default_lang
        self.supported_languages = list(UI_TRANSLATIONS.keys())

    def get_available_languages(self) -> Dict[str, str]:
        """Returns a dictionary of available language codes and their native names."""
        return {lang: UI_TRANSLATIONS[lang]['language_name'] for lang in self.supported_languages}

    def translate_ui(self, key: str, lang: Optional[str] = None) -> str:
        """
        Translates a UI key to the specified language.
        
        Args:
            key: The translation key for a UI element.
            lang: The target language code. Uses the default if not specified.
            
        Returns:
            The translated string, or the key itself if no translation is found.
        """
        target_lang = lang or self.default_lang
        # Use .get() for safe dictionary access, falling back to the key itself.
        return UI_TRANSLATIONS.get(target_lang, {}).get(key, key)

    def translate_disease(self, disease_key: str, lang: Optional[str] = None) -> str:
        """
        Translates a disease name using its unique key.
        
        Args:
            disease_key: The unique key for the disease (e.g., 'Apple___Apple_scab').
            lang: The target language code.
            
        Returns:
            The translated disease name, or the key itself if not found.
        """
        target_lang = lang or self.default_lang
        # Use .get() to safely access the translation, falling back to the key.
        return DISEASE_TRANSLATIONS.get(disease_key, {}).get(target_lang, disease_key)


# ==============================================================================
# DEFAULT INSTANCE
# ==============================================================================

# Create a single, default instance for easy import and use across the application.
translator = Translator()
