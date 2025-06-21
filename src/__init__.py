"""
KrishiSahayak - AI-Powered Crop Health Assistant

A comprehensive AI solution for plant disease classification and agricultural advisory.
"""

__version__ = "0.1.0"
__author__ = "Vikas Sahani"
__email__ = "vikassahani17@gmail.com"
__github__ = "https://github.com/VIKAS9793/KrishiSahayak"

# Import key modules for easier access
from .data.dataset import PlantDiseaseDataset
from .models.model import PlantDiseaseModel
from .models.hybrid import HybridModel

# Set up logging
import logging
from .utils.logger import setup_logging, get_logger

# Initialize logging with default configuration
setup_logging()
logger = get_logger(__name__)
