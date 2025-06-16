# KrishiRakshak – AI-Powered Crop Health Guardian
स्वस्थ फसल, समृद्ध किसान | Healthy Crops, Prosperous Farmers

<p align="center">
  <img src="assets/banners/banner.png?raw=true" alt="KrishiRakshak Banner" width="800">
  <br>
  <img src="assets/logos/logo.png?raw=true" alt="KrishiRakshak Logo" width="200">
</p>

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/VIKAS9793/KrishiRakshak-2.0.git
   cd KrishiRakshak-2.0
   ```

## Project Overview

KrishiRakshak is a comprehensive AI solution that combines real-time plant disease classification with actionable agricultural advisory. Our system is designed to help rural farmers quickly identify and manage plant diseases through:

### Key Features
1. **Real-Time Disease Detection**
   - AI-powered disease identification
   - Offline operation in remote areas
   - Accurate classification of 38 plant diseases

2. **Actionable Advisory System**
   - Immediate disease management recommendations
   - Local language support for better understanding
   - Clear treatment guidelines
   - Emergency response protocols

3. **Resource-Efficient Technology**
   - Optimized MobileNetV3-Large architecture
   - Offline-first deployment
   - Efficient model quantization
   - Multi-format export for various devices

### Core Capabilities
- Real-time disease detection
- Actionable advisory recommendations
- Offline-capable operation
- Multi-language support
- Explainable predictions
- Resource-efficient design

### Project Vision
To empower rural farmers with accurate, accessible, and explainable AI-powered crop health monitoring tools that help prevent crop losses and improve agricultural productivity through immediate advisory support.

### Technical Mission
- Provide accurate disease detection
- Maintain model accuracy on low-resource devices
- Enable offline operation in remote areas
- Support multiple languages for better accessibility
- Generate actionable recommendations

## Technical Architecture

KrishiRakshak leverages a modern deep learning stack for efficient plant disease classification:

### Model Architecture
- **Backbone**: MobileNetV3-Large (pretrained on ImageNet)
- **Classifier Head**: Custom layers with dropout and ReLU activation
- **Input Size**: 224x224 pixels (RGB)
- **Output**: 38 plant disease classes

### Training Pipeline
1. Data loading and augmentation
2. Model training with mixed precision
3. Validation and metric computation
4. Model checkpointing and logging

### Deployment Options
- ONNX Runtime for CPU/GPU inference
- TensorFlow Lite for mobile deployment
- TorchScript for production serving

## Project Structure

```
KrishiRakshak/
├── data/
│   ├── processed_data/    # Processed CSVs
│   └── plantvillage/      # Raw images
├── models/                 # Saved model checkpoints
├── logs/                   # Training logs
├── reports/                # Data reports and visualizations
├── src/
│   ├── config.py         # Configuration
│   ├── data/              # Data loading
│   │   └── datamodule.py
│   ├── models/            # Model architecture
│   │   └── plant_model.py
│   └── train.py           # Training script
├── scripts/
│   ├── fix_dataset.py   # Data fixing utilities
│   └── sanity_check.py    # Data validation
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Prerequisites

#### Hardware Requirements

- **CPU**: Minimum 4 cores (8+ recommended)
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Storage**: At least 10GB free space for dataset and models
- **GPU**: Optional but recommended for faster training (NVIDIA GPU with CUDA support)

#### Software Requirements

- Python 3.10 (recommended) or 3.9
- pip (Python package installer)
- CUDA (optional, for GPU acceleration)
  - For GPU support: CUDA 11.8 and compatible drivers
  - For CPU-only: No additional requirements

### 2. Quick Start

For detailed setup instructions, please refer to the [Quick Start Guide](docs/QUICKSTART.md).

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/KrishiRakshak.git
cd KrishiRakshak

# 2. Follow the setup instructions in docs/QUICKSTART.md
# 3. Prepare your dataset (see Data Preparation below)
# 4. Start training
python src/train.py
```

### 3. Data Preparation

Ensure your data is in the following structure:

```
data/
├── processed_data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── plantvillage/
    ├── train/
    │   ├── class1/
    │   └── class2/
    ├── val/
    └── test/
```

### 3. Training

```bash
# Basic training
python src/train.py

# With custom parameters
python src/train.py \
    --model efficientnet_b3 \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001
```

### 4. Monitoring

Monitor training with TensorBoard:

```bash
tensorboard --logdir=logs/
```

## Key Features

- **Modular Code**: Clean separation of data, model, and training logic
- **Reproducibility**: Full experiment tracking with PyTorch Lightning
- **Efficient Training**: Mixed precision and gradient accumulation
- **Data Augmentation**: Extensive image augmentations with Albumentations
- **Class Imbalance Handling**: Weighted sampling and class weights
- **Model Checkpointing**: Automatic saving of best models
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Cosine annealing with warm restarts

## Model Architecture

- **Backbone**: EfficientNet-B3 (pretrained on ImageNet)
- **Classifier**: Custom head with dropout
- **Loss**: Cross-Entropy with class weights
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine Annealing with warm restarts

## Performance

Monitor training progress using TensorBoard. Key metrics include:

- Training/Validation Loss
- Accuracy
- F1 Score
- Precision
- Recall

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
