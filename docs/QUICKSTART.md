# KrishiRakshak – AI-Powered Crop Health Guardian
स्वस्थ फसल, समृद्ध किसान | Healthy Crops, Prosperous Farmers

## Quick Start Guide

Welcome to KrishiRakshak - your lightweight, resource-efficient solution for plant disease classification. This guide will help you set up and deploy our AI-powered system for rural farmers.

### Project Overview
KrishiRakshak is designed to:
- Run efficiently on low-resource devices
- Provide accurate disease classification
- Support offline deployment
- Generate explainable predictions
- Work with local languages

### Technical Features
- MobileNetV3-Large architecture
- Quantized model support
- Multi-format export (ONNX, TorchScript, TFLite)
- Comprehensive evaluation metrics
- Resource-optimized training
- Offline-first deployment

This guide will help you set up the development environment and get started with the KrishiRakshak project.

## Prerequisites

- Python 3.10 (recommended) or 3.9
- pip (Python package installer)
- CUDA (optional, for GPU acceleration)

## Dataset Setup

### Source
The project uses the PlantVillage dataset available on Kaggle:
- **Source**: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation**: A. Prabhu, A. Singh, M. Singh, and A. Singh, "PlantVillage Dataset - Leaf Images with Disease Information", figshare, 2020. [Online]. Available: https://doi.org/10.6084/m9.figshare.15124244.v1

### Dataset Structure
The dataset contains 54,306 images of plant leaves with 38 different classes:
- 14 healthy plant classes
- 24 diseased plant classes
- Images are in RGB format
- Original images are of varying sizes

#### Important Note on Dataset Usage
This dataset is primarily used for training and demonstrating the functionality of the KrishiRakshak system. For real-world deployment, we recommend:

1. **Field Data Collection**
   - Collect additional images from actual agricultural fields
   - Capture images under different lighting conditions
   - Include various angles and distances
   - Document seasonal variations

2. **Data Diversity**
   - Include images from different geographical regions
   - Capture different stages of disease progression
   - Include images with varying background complexity
   - Document different environmental conditions

3. **Real-World Considerations**
   - Images should be captured using standard smartphone cameras
   - Include images with partial occlusions
   - Document different soil types and field conditions
   - Capture images at different times of day

For production deployment, we recommend creating a custom dataset that:
- Matches your specific geographical region
- Includes local plant species and diseases
- Captures real-world field conditions
- Includes seasonal variations
- Has proper labeling and validation
- Follows best practices for data collection and processing

### Best Practices for Dataset Creation

#### 1. Data Collection
- Use standard smartphone cameras with at least 12MP resolution
- Capture images in both portrait and landscape orientations
- Include images from multiple angles (top, side, close-up)
- Document lighting conditions (sunlight, shade, artificial light)
- Capture images at different times of day (morning, noon, evening)
- Include images with partial occlusions and varying backgrounds
- Document weather conditions (rainy, sunny, cloudy)
- Capture images at different growth stages

#### 2. Data Labeling Guidelines
- Use hierarchical labeling (e.g., Plant_Type/Health_Status/Disease_Type)
- Include confidence scores for ambiguous cases
- Document image metadata (date, time, location, weather)
- Use multiple labelers for consistency
- Implement quality control checks
- Maintain detailed labeling guidelines
- Document edge cases and exceptions

#### 3. Data Validation
- Split dataset into train/validation/test sets (70/15/15)
- Use stratified sampling to maintain class distribution
- Implement cross-validation for model evaluation
- Document distribution of classes in each split
- Track image quality metrics
- Maintain version control of labeled data
- Regularly audit and update labels

#### 4. Data Augmentation Best Practices
- Basic Augmentations:
  - Random rotation (±20 degrees)
  - Horizontal and vertical flips
  - Random crops and resizing
  - Color jitter (brightness, contrast, saturation)
  - Gaussian noise addition
  - Random erasing

- Advanced Augmentations:
  - MixUp and CutMix for synthetic samples
  - Style transfer for domain adaptation
  - Weather effects simulation
  - Lighting condition variations
  - Background augmentation
  - Perspective transformations

- Augmentation Guidelines:
  - Maintain class balance
  - Preserve disease characteristics
  - Avoid artifacts
  - Document augmentation parameters
  - Monitor impact on model performance
  - Use domain-specific augmentations

### Setup Instructions

1. **Download the Dataset**
   ```bash
   # Download from Kaggle
   kaggle datasets download -d abdallahalidev/plantvillage-dataset
   unzip plantvillage-dataset.zip -d data/
   ```

2. **Dataset Directory Structure**
   After downloading and extracting, the dataset should be organized as:
   ```
   data/
   ├── train/
   │   ├── Apple___Apple_scab/
   │   ├── Apple___Black_rot/
   │   ├── Apple___Cedar_apple_rust/
   │   ├── Apple___healthy/
   │   └── ...
   └── test/
       ├── Apple___Apple_scab/
       ├── Apple___Black_rot/
       ├── Apple___Cedar_apple_rust/
       ├── Apple___healthy/
       └── ...
   ```

3. **Data Augmentation**
   The training pipeline includes the following augmentations:
   - Random horizontal and vertical flips
   - Random rotation (20 degrees)
   - Color jitter (brightness, contrast, saturation)
   - Image resizing to 224x224
   - Normalization using ImageNet statistics

## Setup Instructions

### Windows

1. **Create and Activate Virtual Environment**

   ```powershell
   # Create a virtual environment
   python -m venv .venv
   
   # Activate the virtual environment
   # In PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Or in Command Prompt:
   # .\.venv\Scripts\activate.bat
   ```

2. **Install Dependencies**

   ```powershell
   # Upgrade pip
   python -m pip install --upgrade pip
   
   # Install PyTorch with CUDA 11.8 (adjust CUDA version if needed)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Install remaining dependencies
   pip install -r requirements.txt
   ```

### Linux/macOS

1. **Create and Activate Virtual Environment**

   ```bash
   # Create virtual environment
   python3 -m venv .venv
   
   # Activate virtual environment
   source .venv/bin/activate
   ```

2. **Install Dependencies**

   ```bash
   # Upgrade pip
   python -m pip install --upgrade pip
   
   # Install PyTorch with CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Install remaining dependencies
   pip install -r requirements.txt
   ```

### Using Conda (Alternative)

```bash
# Create and activate conda environment
conda create -n krishirakshak python=3.10
conda activate krishirakshak

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10.x

# Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Next Steps

1. Prepare your dataset following the [data preparation guide](../docs/DATA_PREPARATION.md)
2. Start training your model:
   ```bash
   python src/train.py
   ```
3. Monitor training with TensorBoard:
   ```bash
   tensorboard --logdir=logs/
   ```

## Troubleshooting

- **CUDA not available**: Make sure you have the correct CUDA version installed and your GPU drivers are up to date.
- **Package conflicts**: Use a fresh virtual environment to avoid conflicts.
- **Missing dependencies**: Run `pip install -r requirements.txt` to ensure all dependencies are installed.

For additional help, please refer to the [documentation](../docs/) or open an issue in the repository.
