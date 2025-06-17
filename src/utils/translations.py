"""Translation module for multilingual support."""
from typing import Dict, List, Optional

# Translation dictionaries for supported languages
TRANSLATIONS = {
    'en': {
        'language_name': 'English',
        'upload_image': 'Upload an image of a plant leaf',
        'predict': 'Predict',
        'clear': 'Clear',
        'prediction': 'Prediction',
        'confidence': 'Confidence',
        'no_prediction': 'No prediction available',
        'error': 'Error',
        'error_processing': 'Error processing image',
    },
    'hi': {
        'language_name': 'हिंदी',
        'upload_image': 'पौधे की पत्ती की एक तस्वीर अपलोड करें',
        'predict': 'भविष्यवाणी करें',
        'clear': 'साफ़ करें',
        'prediction': 'भविष्यवाणी',
        'confidence': 'विश्वास',
        'no_prediction': 'कोई भविष्यवाणी उपलब्ध नहीं',
        'error': 'त्रुटि',
        'error_processing': 'छवि प्रसंस्करण में त्रुटि',
    },
    'mr': {
        'language_name': 'मराठी',
        'upload_image': 'वनस्पतीच्या पानाची प्रतिमा अपलोड करा',
        'predict': 'अंदाज करा',
        'clear': 'साफ करा',
        'prediction': 'अंदाज',
        'confidence': 'आत्मविश्वास',
        'no_prediction': 'कोणताही अंदाज उपलब्ध नाही',
        'error': 'त्रुटी',
        'error_processing': 'प्रतिमा प्रक्रिया करताना त्रुटी',
    }
}

# Disease name translations (English to other languages)
DISEASE_TRANSLATIONS = {
    'Apple___Apple_scab': {
        'en': 'Apple Scab',
        'hi': 'सेब का खरोंच',
        'mr': 'सफरचंद खरुज'
    },
    'Apple___Black_rot': {
        'en': 'Apple Black Rot',
        'hi': 'सेब का काला सड़न',
        'mr': 'सफरचंद काळा कुजवा'
    },
    # Add more disease translations as needed
}

class Translator:
    """Simple translation service for multilingual support."""
    
    def __init__(self, default_lang: str = 'en'):
        """
        Initialize translator.
        
        Args:
            default_lang: Default language code (e.g., 'en', 'hi', 'mr')
        """
        self.default_lang = default_lang
        self.supported_languages = list(TRANSLATIONS.keys())
    
    def get_available_languages(self) -> Dict[str, str]:
        """Get dictionary of available language codes and names."""
        return {
            lang: TRANSLATIONS[lang]['language_name']
            for lang in self.supported_languages
        }
    
    def translate(self, key: str, lang: Optional[str] = None) -> str:
        """
        Translate a key to the specified language.
        
        Args:
            key: Translation key
            lang: Target language code. Uses default if not specified.
            
        Returns:
            Translated string or the key if translation not found
        """
        lang = lang or self.default_lang
        try:
            return TRANSLATIONS.get(lang, {}).get(key, key)
        except Exception:
            return key
    
    def translate_disease(self, disease_key: str, lang: Optional[str] = None) -> str:
        """
        Translate a disease name.
        
        Args:
            disease_key: Disease key (e.g., 'Apple___Apple_scab')
            lang: Target language code. Uses default if not specified.
            
        Returns:
            Translated disease name or original key if not found
        """
        lang = lang or self.default_lang
        try:
            return DISEASE_TRANSLATIONS.get(disease_key, {}).get(lang, disease_key)
        except Exception:
            return disease_key

# Create a default translator instance
translator = Translator()
