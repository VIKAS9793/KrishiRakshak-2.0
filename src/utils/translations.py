"""Translation module for multilingual support."""
from typing import Dict, Optional

# It's recommended to move these dictionaries to separate files (e.g., JSON)
# for better scalability and maintainability.
TRANSLATIONS = {
    'en': {
        'language_name': 'English',
        'upload_image': 'Upload an image of a plant leaf',
        'predict': 'Predict',
        'clear': 'Clear',
        'prediction': 'Prediction',
    },
    'hi': {
        'language_name': 'हिंदी',
        'upload_image': 'पौधे की पत्ती की एक तस्वीर अपलोड करें',
        'predict': 'भविष्यवाणी करें',
        'clear': 'साफ़ करें',
        'prediction': 'भविष्यवाणी',
    },
    'mr': {
        'language_name': 'मराठी',
        'upload_image': 'वनस्पतीच्या पानाची प्रतिमा अपलोड करा',
        'predict': 'अंदाज करा',
        'clear': 'साफ करा',
        'prediction': 'अंदाज',
    }
}

DISEASE_TRANSLATIONS = {
    'Apple___Apple_scab': {
        'en': 'Apple Scab', 'hi': 'सेब का खरोंच', 'mr': 'सफरचंद खरुज'
    },
    'Apple___Black_rot': {
        'en': 'Apple Black Rot', 'hi': 'सेब का काला सड़न', 'mr': 'सफरचंद काळा कुजवा'
    },
}

class Translator:
    """A simple translation service for providing multilingual support."""
    
    def __init__(self, default_lang: str = 'en'):
        """
        Initializes the translator.
        
        Args:
            default_lang: The default language code (e.g., 'en', 'hi').
        """
        self.default_lang = default_lang
        self.supported_languages = list(TRANSLATIONS.keys())

    def get_available_languages(self) -> Dict[str, str]:
        """Returns a dictionary of available language codes and their names."""
        return {lang: TRANSLATIONS[lang]['language_name'] for lang in self.supported_languages}

    def _get_translation(self, source_dict: Dict, main_key: str, lang: str, fallback: str) -> str:
        """Private helper to get a translation from a source dictionary."""
        return source_dict.get(main_key, {}).get(lang, fallback)

    def translate(self, key: str, lang: Optional[str] = None) -> str:
        """
        Translates a UI key to the specified language.
        
        Args:
            key: The translation key for a UI element.
            lang: The target language code. Uses the default if not specified.
            
        Returns:
            The translated string, or the key itself if no translation is found.
        """
        target_lang = lang or self.default_lang
        # Use .get() for safe dictionary access without try/except
        return TRANSLATIONS.get(target_lang, {}).get(key, key)

    def translate_disease(self, disease_key: str, lang: Optional[str] = None) -> str:
        """
        Translates a disease name.
        
        Args:
            disease_key: The unique key for the disease (e.g., 'Apple___Apple_scab').
            lang: The target language code.
            
        Returns:
            The translated disease name, or the original key if not found.
        """
        target_lang = lang or self.default_lang
        return self._get_translation(DISEASE_TRANSLATIONS, disease_key, target_lang, disease_key)

# A default translator instance for easy import and use across the application.
translator = Translator()
