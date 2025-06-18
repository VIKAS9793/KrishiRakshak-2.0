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
        # App Info
        'app_title': 'KrishiRakshak - Plant Health Assistant',
        'app_description': 'AI-powered plant disease detection and advisory system',
        'language_name': 'English',
        
        # Navigation
        'home': 'Home',
        'about': 'About',
        'help': 'Help',
        'settings': 'Settings',
        
        # Main UI
        'upload_image': 'Upload an image of a plant leaf',
        'capture_image': 'Capture using camera',
        'or': 'OR',
        'predict': 'Analyze Plant Health',
        'clear': 'Clear All',
        'back': 'Back',
        'next': 'Next',
        'submit': 'Submit',
        'cancel': 'Cancel',
        'save': 'Save Changes',
        'reset': 'Reset to Default',
        
        # Results
        'prediction': 'Disease Detected',
        'confidence': 'Confidence Level',
        'heatmap': 'Affected Areas',
        'no_disease': 'No disease detected',
        'healthy_plant': 'Plant appears healthy',
        'prediction_accuracy': 'Model Confidence',
        'analysis': 'Analysis Results',
        'details': 'View Details',
        
        # Advisory
        'prediction_tab': 'Disease Analysis',
        'advisory_tab': 'Treatment Plan',
        'prevention_tab': 'Prevention Tips',
        'upload_image_for_advice': 'Upload an image to get personalized plant health advice',
        'symptoms': 'Key Symptoms',
        'prevention': 'Preventive Measures',
        'organic_treatment': 'Organic Solutions',
        'chemical_treatment': 'Chemical Controls',
        'recommended_products': 'Recommended Products',
        'application_method': 'Application Method',
        'frequency': 'Treatment Frequency',
        'precautions': 'Safety Precautions',
        'source': 'Reference Sources',
        'disclaimer': 'Note: For best results, consult with a local agricultural expert',
        
        # Status Messages
        'loading': 'Analyzing plant health...',
        'processing': 'Processing image...',
        'success': 'Analysis complete!',
        'error': 'Error Occurred',
        'no_image': 'Please upload or capture an image to begin analysis.',
        'image_too_large': 'Image is too large. Maximum size is 5MB.',
        'invalid_format': 'Unsupported image format. Please use JPG, JPEG, or PNG.',
        'prediction_error': 'Error during analysis. Please try again with a clearer image.',
        'network_error': 'Network connection error. Please check your internet connection.',
        'try_again': 'Try Again',
        
        # Settings
        'language': 'Select Language',
        'theme': 'Theme',
        'dark_mode': 'Dark Mode',
        'notifications': 'Notifications',
        'data_usage': 'Data Usage',
        'privacy': 'Privacy Settings',
        'terms': 'Terms of Service',
        'privacy_policy': 'Privacy Policy',
        'version': 'App Version',
        'check_updates': 'Check for Updates',
        'clear_cache': 'Clear Cache',
        'logout': 'Log Out',
    },
    'hi': {
        # App Info
        'app_title': 'कृषि रक्षक - पौध स्वास्थ्य सहायक',
        'app_description': 'कृत्रिम बुद्धिमत्ता आधारित पौध रोग पहचान एवं सलाह प्रणाली',
        'language_name': 'हिंदी',
        
        # Navigation
        'home': 'मुख्य पृष्ठ',
        'about': 'हमारे बारे में',
        'help': 'सहायता',
        'settings': 'सेटिंग्स',
        
        # Main UI
        'upload_image': 'पौधे की पत्ती की तस्वीर अपलोड करें',
        'capture_image': 'कैमरा से तस्वीर लें',
        'or': 'अथवा',
        'predict': 'पौधे का स्वास्थ्य जांचें',
        'clear': 'सभी साफ करें',
        'back': 'पीछे जाएं',
        'next': 'आगे बढ़ें',
        'submit': 'जमा करें',
        'cancel': 'रद्द करें',
        'save': 'परिवर्तन सहेजें',
        'reset': 'डिफ़ॉल्ट पर रीसेट करें',
        
        # Results
        'prediction': 'पहचाना गया रोग',
        'confidence': 'विश्वसनीयता स्तर',
        'heatmap': 'प्रभावित क्षेत्र',
        'no_disease': 'कोई रोग पहचान में नहीं आया',
        'healthy_plant': 'पौधा स्वस्थ प्रतीत होता है',
        'prediction_accuracy': 'मॉडल की विश्वसनीयता',
        'analysis': 'विश्लेषण परिणाम',
        'details': 'विस्तार से देखें',
        
        # Advisory
        'prediction_tab': 'रोग विश्लेषण',
        'advisory_tab': 'उपचार योजना',
        'prevention_tab': 'रोकथाम के उपाय',
        'upload_image_for_advice': 'व्यक्तिगत पौध स्वास्थ्य सलाह के लिए एक छवि अपलोड करें',
        'symptoms': 'मुख्य लक्षण',
        'prevention': 'निवारक उपाय',
        'organic_treatment': 'जैविक समाधान',
        'chemical_treatment': 'रासायनिक नियंत्रण',
        'recommended_products': 'सुझाए गए उत्पाद',
        'application_method': 'आवेदन विधि',
        'frequency': 'उपचार आवृत्ति',
        'precautions': 'सावधानियां',
        'source': 'संदर्भ स्रोत',
        'disclaimer': 'नोट: सर्वोत्तम परिणामों के लिए, किसी स्थानीय कृषि विशेषज्ञ से परामर्श करें',
        
        # Status Messages
        'loading': 'पौधे के स्वास्थ्य का विश्लेषण किया जा रहा है...',
        'processing': 'छवि प्रोसेस की जा रही है...',
        'success': 'विश्लेषण पूर्ण हुआ!',
        'error': 'त्रुटि हुई',
        'no_image': 'विश्लेषण शुरू करने के लिए कृपया कोई छवि अपलोड करें या कैमरे से कैप्चर करें।',
        'image_too_large': 'छवि का आकार बहुत बड़ा है। अधिकतम आकार 5MB होना चाहिए।',
        'invalid_format': 'असमर्थित छवि प्रारूप। कृपया JPG, JPEG, या PNG प्रारूप का उपयोग करें।',
        'prediction_error': 'विश्लेषण के दौरान त्रुटि हुई। कृपया स्पष्ट छवि के साथ पुनः प्रयास करें।',
        'network_error': 'नेटवर्क कनेक्शन त्रुटि। कृपया अपना इंटरनेट कनेक्शन जांचें।',
        'try_again': 'पुनः प्रयास करें',
        
        # Settings
        'language': 'भाषा चुनें',
        'theme': 'थीम',
        'dark_mode': 'डार्क मोड',
        'notifications': 'सूचनाएं',
        'data_usage': 'डेटा उपयोग',
        'privacy': 'गोपनीयता सेटिंग्स',
        'terms': 'सेवा की शर्तें',
        'privacy_policy': 'गोपनीयता नीति',
        'version': 'ऐप संस्करण',
        'check_updates': 'अपडेट के लिए जांचें',
        'clear_cache': 'कैश साफ़ करें',
        'logout': 'लॉग आउट',
    },
    'mr': {
        # App Info
        'app_title': 'कृषीरक्षक - वनस्पती आरोग्य सहाय्यक',
        'app_description': 'कृत्रिम बुद्धिमत्ता आधारित वनस्पती रोग ओळख आणि सल्ला प्रणाली',
        'language_name': 'मराठी',
        
        # Navigation
        'home': 'मुख्यपृष्ठ',
        'about': 'आमच्याबद्दल',
        'help': 'मदत',
        'settings': 'सेटिंग्ज',
        
        # Main UI
        'upload_image': 'वनस्पतीच्या पानाची प्रतिमा अपलोड करा',
        'capture_image': 'कॅमेर्यावरुन प्रतिमा कॅप्चर करा',
        'or': 'किंवा',
        'predict': 'वनस्पतीच्या आरोग्याचे विश्लेषण करा',
        'clear': 'सर्व साफ करा',
        'back': 'मागे जा',
        'next': 'पुढे',
        'submit': 'सबमिट करा',
        'cancel': 'रद्द करा',
        'save': 'बदल जतन करा',
        'reset': 'डीफॉल्टवर रीसेट करा',
        
        # Results
        'prediction': 'ओळखलेला रोग',
        'confidence': 'आत्मविश्वास पातळी',
        'heatmap': 'प्रभावित क्षेत्र',
        'no_disease': 'कोणताही रोग आढळला नाही',
        'healthy_plant': 'वनस्पती निरोगी दिसते',
        'prediction_accuracy': 'मॉडेलची अचूकता',
        'analysis': 'विश्लेषण परिणाम',
        'details': 'तपशील पहा',
        
        # Advisory
        'prediction_tab': 'रोग विश्लेषण',
        'advisory_tab': 'उपचार योजना',
        'prevention_tab': 'प्रतिबंधात्मक उपाय',
        'upload_image_for_advice': 'वैयक्तिकृत वनस्पती आरोग्य सल्ल्यासाठी प्रतिमा अपलोड करा',
        'symptoms': 'मुख्य लक्षणे',
        'prevention': 'प्रतिबंधात्मक उपाय',
        'organic_treatment': 'सेंद्रिय उपाय',
        'chemical_treatment': 'रासायनिक नियंत्रण',
        'recommended_products': 'शिफारस केलेले उत्पादने',
        'application_method': 'अर्ज पद्धत',
        'frequency': 'उपचार वारंवारता',
        'precautions': 'सावधगिरी',
        'source': 'संदर्भ स्रोत',
        'disclaimer': 'टीप: सर्वोत्तम परिणामांसाठी, स्थानिक शेती तज्ञांचा सल्ला घ्या',
        
        # Status Messages
        'loading': 'वनस्पतीच्या आरोग्याचे विश्लेषण केले जात आहे...',
        'processing': 'प्रतिमा प्रक्रिया करत आहे...',
        'success': 'विश्लेषण पूर्ण झाले!',
        'error': 'त्रुटी आढळली',
        'no_image': 'कृपया विश्लेषण सुरू करण्यासाठी प्रतिमा अपलोड करा किंवा कॅमेर्यावरुन कॅप्चर करा.',
        'image_too_large': 'प्रतिमा खूप मोठी आहे. कमाल आकार 5MB पेक्षा कमी असावा.',
        'invalid_format': 'असमर्थित प्रतिमा स्वरूप. कृपया JPG, JPEG, किंवा PNG वापरा.',
        'prediction_error': 'विश्लेषणादरम्यान त्रुटी. कृपया स्पष्ट प्रतिमेसह पुन्हा प्रयत्न करा.',
        'network_error': 'नेटवर्क कनेक्शन त्रुटी. कृपया आपले इंटरनेट कनेक्शन तपासा.',
        'try_again': 'पुन्हा प्रयत्न करा',
        
        # Settings
        'language': 'भाषा निवडा',
        'theme': 'थीम',
        'dark_mode': 'डार्क मोड',
        'notifications': 'सूचना',
        'data_usage': 'डेटा वापर',
        'privacy': 'गोपनीयता सेटिंग्ज',
        'terms': 'सेवा अटी',
        'privacy_policy': 'गोपनीयता धोरण',
        'version': 'अॅप आवृत्ती',
        'check_updates': 'अद्ययावत् तपासा',
        'clear_cache': 'कॅशे साफ करा',
        'logout': 'बाहेर जा',
    }
}

# --- Disease Name Translations ---
DISEASE_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # Apple
    'Apple___Apple_scab': {
        'en': 'Apple Scab (Venturia inaequalis)',
        'hi': 'सेब का खरोंच रोग (वेंच्युरिया इनएक्वालिस)',
        'mr': 'सफरचंद खरुज (व्हेंच्युरिया इनएक्वालिस)'
    },
    'Apple___Black_rot': {
        'en': 'Apple Black Rot (Botryosphaeria obtusa)',
        'hi': 'सेब का काला सड़न रोग (बोट्रीओस्फेरिया ऑब्ट्यूसा)',
        'mr': 'सफरचंद काळा कुजवा (बोट्रिओस्फेरिया ऑब्ट्यूसा)'
    },
    'Apple___Cedar_apple_rust': {
        'en': 'Cedar Apple Rust (Gymnosporangium juniperi-virginianae)',
        'hi': 'देवदार सेब की जंग (जिम्नोस्पोरेंजियम जुनिपेरी-वर्जिनियाने)',
        'mr': 'देवदार सफरचंद गंज (जिम्नोस्पोरेंजियम जुनिपेरी-व्हर्जिनियाने)'
    },
    'Apple___healthy': {
        'en': 'Healthy Apple',
        'hi': 'स्वस्थ सेब',
        'mr': 'निरोगी सफरचंद'
    },
    
    # Cherry
    'Cherry_(including_sour)___Powdery_mildew': {
        'en': 'Cherry Powdery Mildew (Podosphaera clandestina)',
        'hi': 'चेरी पाउडर फफूंद (पोडोस्फेरा क्लैंडेस्टिना)',
        'mr': 'चेरी पावडरी मिल्ड्यू (पोडोस्फेरा क्लॅन्डेस्टिना)'
    },
    'Cherry_(including_sour)___healthy': {
        'en': 'Healthy Cherry',
        'hi': 'स्वस्थ चेरी',
        'mr': 'निरोगी चेरी'
    },
    
    # Corn (Maize)
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'en': 'Corn Gray Leaf Spot (Cercospora zeae-maydis)',
        'hi': 'मक्के का ग्रे पत्ती धब्बा रोग (सेरकोस्पोरा ज़ी-मेडिस)',
        'mr': 'मक्यावरील करडे पानांचे डाग (सेरकोस्पोरा झी-मेडिस)'
    },
    'Corn_(maize)___Common_rust_': {
        'en': 'Corn Common Rust (Puccinia sorghi)',
        'hi': 'मक्के का सामान्य जंग (पक्किनिया सोरघी)',
        'mr': 'मक्याचा सामान्य गंज (पुक्किनिया सोरघी)'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'en': 'Corn Northern Leaf Blight (Exserohilum turcicum)',
        'hi': 'मक्के का उत्तरी पत्ती झुलसा (एक्ससेरोहिलम टर्सिकम)',
        'mr': 'मक्याचा उत्तर पानांचा झटका (एक्ससेरोहिलम टर्सिकम)'
    },
    'Corn_(maize)___healthy': {
        'en': 'Healthy Corn',
        'hi': 'स्वस्थ मक्का',
        'mr': 'निरोगी मका'
    },
    
    # Grape
    'Grape___Black_rot': {
        'en': 'Grape Black Rot (Guignardia bidwellii)',
        'hi': 'अंगूर का काला सड़न (गिग्नार्डिया बिडवेल्ली)',
        'mr': 'द्राक्ष काळा कुजवा (गिग्नार्डिया बिडवेल्ली)'
    },
    'Grape___Esca_(Black_Measles)': {
        'en': 'Grape Esca (Phaeomoniella spp.)',
        'hi': 'अंगूर एस्का (फियोमोनिएला एसपीपी.)',
        'mr': 'द्राक्ष एस्का (फियोमोनिएला एसपीपी.)'
    },
    'Grape___healthy': {
        'en': 'Healthy Grape',
        'hi': 'स्वस्थ अंगूर',
        'mr': 'निरोगी द्राक्ष'
    },
    
    # Orange
    'Orange___Haunglongbing_(Citrus_greening)': {
        'en': 'Citrus Greening (Candidatus Liberibacter spp.)',
        'hi': 'सिट्रस ग्रीनिंग (कैंडिडेटस लिबेरिबैक्टर एसपीपी.)',
        'mr': 'लिंबूवर्गीय हरितीकरण (कॅन्डिडॅटस लिबेरिबॅक्टर एसपीपी.)'
    },
    
    # Peach
    'Peach___Bacterial_spot': {
        'en': 'Peach Bacterial Spot (Xanthomonas arboricola)',
        'hi': 'आड़ू का जीवाणु धब्बा (जैंथोमोनास आर्बोरिकोला)',
        'mr': 'सुदंर आंबा जिवाणू ठिपके (झॅन्थोमोनास आर्बोरिकोला)'
    },
    'Peach___healthy': {
        'en': 'Healthy Peach',
        'hi': 'स्वस्थ आड़ू',
        'mr': 'निरोगी सुदंर आंबा'
    },
    
    # Bell Pepper
    'Pepper,_bell___Bacterial_spot': {
        'en': 'Bell Pepper Bacterial Spot (Xanthomonas spp.)',
        'hi': 'शिमला मिर्च का जीवाणु धब्बा (जैंथोमोनास एसपीपी.)',
        'mr': 'भोपळी मिरची जिवाणू ठिपके (झॅन्थोमोनास एसपीपी.)'
    },
    'Pepper,_bell___healthy': {
        'en': 'Healthy Bell Pepper',
        'hi': 'स्वस्थ शिमला मिर्च',
        'mr': 'निरोगी भोपळी मिरची'
    },
    
    # Potato
    'Potato___Early_blight': {
        'en': 'Potato Early Blight (Alternaria solani)',
        'hi': 'आलू का अगेती झुलसा (अल्टरनेरिया सोलानी)',
        'mr': 'बटाटा लवकर येणारा झटका (अल्टरनेरिया सोलानी)'
    },
    'Potato___Late_blight': {
        'en': 'Potato Late Blight (Phytophthora infestans)',
        'hi': 'आलू का पछेती झुलसा (फाइटोफ्थोरा इन्फेस्टन्स)',
        'mr': 'बटाटा उशिरा येणारा झटका (फायटोफ्थोरा इन्फेस्टन्स)'
    },
    'Potato___healthy': {
        'en': 'Healthy Potato',
        'hi': 'स्वस्थ आलू',
        'mr': 'निरोगी बटाटा'
    },
    
    # Strawberry
    'Strawberry___Leaf_scorch': {
        'en': 'Strawberry Leaf Scorch (Diplocarpon earlianum)',
        'hi': 'स्ट्रॉबेरी पत्ती झुलसा (डिप्लोकार्पन अर्लियानम)',
        'mr': 'स्ट्रॉबेरी पाने जळणे (डिप्लोकार्पॉन अर्लियानम)'
    },
    'Strawberry___healthy': {
        'en': 'Healthy Strawberry',
        'hi': 'स्वस्थ स्ट्रॉबेरी',
        'mr': 'निरोगी स्ट्रॉबेरी'
    },
    
    # Tomato
    'Tomato___Bacterial_spot': {
        'en': 'Tomato Bacterial Spot (Xanthomonas spp.)',
        'hi': 'टमाटर का जीवाणु धब्बा (जैंथोमोनास एसपीपी.)',
        'mr': 'टोमॅटो जिवाणू ठिपके (झॅन्थोमोनास एसपीपी.)'
    },
    'Tomato___Early_blight': {
        'en': 'Tomato Early Blight (Alternaria solani)',
        'hi': 'टमाटर का अगेती झुलसा (अल्टरनेरिया सोलानी)',
        'mr': 'टोमॅटो लवकर येणारा झटका (अल्टरनेरिया सोलानी)'
    },
    'Tomato___Late_blight': {
        'en': 'Tomato Late Blight (Phytophthora infestans)',
        'hi': 'टमाटर का पछेती झुलसा (फाइटोफ्थोरा इन्फेस्टन्स)',
        'mr': 'टोमॅटो उशिरा येणारा झटका (फायटोफ्थोरा इन्फेस्टन्स)'
    },
    'Tomato___Leaf_Mold': {
        'en': 'Tomato Leaf Mold (Fulvia fulva)',
        'hi': 'टमाटर की पत्ती फफूंद (फुल्विया फुल्वा)',
        'mr': 'टोमॅटो पानांची बुरशी (फुल्विया फुल्वा)'
    },
    'Tomato___Septoria_leaf_spot': {
        'en': 'Tomato Septoria Leaf Spot (Septoria lycopersici)',
        'hi': 'टमाटर सेप्टोरिया पत्ती धब्बा (सेप्टोरिया लाइकोपर्सिकी)',
        'mr': 'टोमॅटो सेप्टोरिया पानांचे डाग (सेप्टोरिया लायकोपर्सिसी)'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'en': 'Tomato Spider Mites (Tetranychus urticae)',
        'hi': 'टमाटर मकड़ी कीट (टेट्रानाइकस अर्टिके)',
        'mr': 'टोमॅटो स्पायडर माईट (टेट्रानायकस युर्टिके)'
    },
    'Tomato___Target_Spot': {
        'en': 'Tomato Target Spot (Corynespora cassiicola)',
        'hi': 'टमाटर टारगेट स्पॉट (कोरिनेस्पोरा कैसीकोला)',
        'mr': 'टोमॅटो टार्गेट स्पॉट (कोरिनेस्पोरा कॅसिकोला)'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'en': 'Tomato Yellow Leaf Curl Virus (TYLCV)',
        'hi': 'टमाटर पीला पत्ती कर्ल वायरस (TYLCV)',
        'mr': 'टोमॅटो पिवळ्या पानांचा कर्ल विषाणू (TYLCV)'
    },
    'Tomato___Tomato_mosaic_virus': {
        'en': 'Tomato Mosaic Virus (ToMV)',
        'hi': 'टमाटर मोज़ेक वायरस (ToMV)',
        'mr': 'टोमॅटो मोझेक विषाणू (ToMV)'
    },
    'Tomato___healthy': {
        'en': 'Healthy Tomato',
        'hi': 'स्वस्थ टमाटर',
        'mr': 'निरोगी टोमॅटो'
    },
    
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
