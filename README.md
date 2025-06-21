# KrishiSahayak – AI-Powered Crop Health Assistant
स्वस्थ फसल, समृद्ध किसान | Healthy Crops, Prosperous Farmers

## 🌱 Hybrid RGB-Multispectral Plant Disease Classification

KrishiSahayak features an advanced hybrid model that combines RGB and multispectral (MS) imaging for more accurate plant disease classification. This enhancement provides improved diagnostic capabilities, especially in challenging lighting conditions.

## 📦 Project Structure

```
KrishiSahayak/
├── configs/               # Configuration files
├── data/                  # Dataset directory (gitignored)
│   └── plantvillage/     # Processed dataset
│       ├── processed_images/  # Preprocessed images
│       ├── train.csv       # Training metadata
│       ├── val.csv         # Validation metadata
│       └── test.csv        # Test metadata
├── models/                # Trained models and artifacts
├── scripts/               # Utility scripts
├── src/                   # Source code
│   ├── models/           # Model architectures
│   └── utils/            # Utility functions
└── tests/                # Test files
```

## 🏗️ Model Architecture

The project uses a hybrid model architecture that combines:

1. **Feature Extraction**: Pre-trained CNN backbone (EfficientNet-B3 by default)
2. **Multi-Scale Feature Fusion**: Combines features from different network depths
3. **Classification Head**: Custom head with dropout and fully connected layers

### 🎯 Performance Metrics

- **Accuracy**: ~98.5% on test set
- **F1-Score**: ~0.984
- **Precision**: ~0.983
- **Recall**: ~0.985

## 📊 Dataset

The dataset is based on the PlantVillage dataset with additional processing:

### Data Processing Pipeline

1. **Image Preprocessing**:
   - Resized to 256x256 pixels
   - Normalized using ImageNet mean and standard deviation
   - Augmented with random flips, rotations, and color jitter

2. **Class Balancing**:
   - Applied class weights to handle imbalanced classes
   - Used oversampling for minority classes

3. **Train/Val/Test Split**:
   - Training: 70%
   - Validation: 15%
   - Test: 15%
   - Stratified split to maintain class distribution

### 🆕 Enhanced Hybrid Model Features

#### Data Handling
- **Dual-Modal Input**: Processes both RGB and multispectral data
- **Synthetic MS Support**: Fallback to synthetic MS data when real MS data is unavailable
- **Robust Preprocessing**: Handles missing or corrupted data gracefully
- **Synchronized Augmentation**: Consistent transformations applied to both RGB and MS data

#### Model Architecture
- **Flexible Backbones**: Supports any timm model architecture
- **Multiple Fusion Strategies**:
  - **Concatenation**: Simple feature concatenation
  - **Addition**: Element-wise feature addition
  - **Attention**: Learnable attention-based fusion
- **Progressive Unfreezing**: Gradually unfreezes MS backbone during training
- **Knowledge Distillation**: Transfer learning from RGB to MS branch

#### Performance Optimizations
- **Mixed Precision Training**: Faster training with FP16/FP32 mixed precision
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Early Stopping**: Prevents overfitting

#### Validation & Monitoring
- **Data Quality Checks**: Validates MS data integrity and consistency
- **Comprehensive Logging**: Tracks training metrics in real-time
- **Visualization**: Feature maps and attention visualization
- **Model Checkpointing**: Saves best models during training

🤖 About the Project
KrishiSahayak is a comprehensive AI solution that combines real-time plant disease classification with actionable agricultural advisory. Built with PyTorch, it is designed to help rural farmers quickly and accurately identify crop diseases using their mobile devices, even in offline environments. By leveraging explainable AI, our system not only provides a diagnosis but also builds trust with users by showing how it arrived at its conclusion.

The project's vision is to empower rural farmers with accessible and accurate crop health tools, helping to prevent crop losses and improve agricultural productivity through immediate, understandable advice.

✨ Key Features

### Core Features
- **Hybrid Model Architecture**: Combines RGB and multispectral data for improved accuracy
- **Synthetic MS Generation**: GAN-based approach to generate synthetic MS data from RGB
- **Knowledge Distillation**: Transfer knowledge from the hybrid model to a lightweight RGB-only model
- **Real-Time Disease Detection**: Accurately identifies 38+ plant diseases across various crops
- **Explainable AI (XAI)**: Implements Grad-CAM to generate visual heatmaps
- **Actionable Advisory**: Provides immediate disease management recommendations
- **Multilingual Support**: Supports English, Hindi (हिंदी), and Marathi (मराठी)
- **Offline-First Design**: Optimized for on-device inference in remote areas
- **Resource-Efficient**: Fast inference on low-resource devices
- **Confidence Scores**: Provides prediction confidence levels
- **Model Export**: Supports PyTorch, ONNX, and TensorFlow Lite formats

🛠️ Technology Stack

### Core Technologies
- **Deep Learning**: PyTorch, PyTorch Lightning
- **Model Architectures**:
  - Hybrid RGB-MS: Custom architecture with EfficientNet backbone
  - Teacher Model: EfficientNetV2
  - Student Model: MobileNetV3-Large (via timm)
- **Data Processing**: Albumentations
- **Web Interface**: Gradio
- **Deployment**: ONNX, TensorFlow Lite, TorchScript
- **Experiment Tracking**: Weights & Biases, TensorBoard
- **Synthetic Data**: GAN-based MS data generation

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or 3.10
- pip (Python package installer)
- CPU: Minimum 4 cores (8+ recommended)
- RAM: Minimum 8GB (16GB+ recommended)
- GPU: Optional but highly recommended for faster training (NVIDIA GPU with CUDA 11.8 support)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/VIKAS9793/KrishiSahayak.git
   cd KrishiSahayak
   ```

2. **Set up a virtual environment** (recommended)
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the package**
   ```bash
   # Install in development mode with all optional dependencies
   pip install -e ".[dev,train,data]"
   
   # Or for production (minimal dependencies)
   # pip install .
   ```

### Training the Model

1. **Prepare the dataset**
   - Download the PlantVillage dataset
   - Place the data in the `data/plantvillage/` directory
   - Run the preprocessing script:
     ```bash
     python scripts/preprocess_dataset.py --input_dir /path/to/raw/data --output_dir data/plantvillage/processed_images
     ```

2. **Start training**
   ```bash
   python scripts/train.py --config configs/train_config.yaml
   ```

3. **Monitor training** with Weights & Biases:
   ```bash
   wandb login
   python scripts/train.py --config configs/train_config.yaml --log_to_wandb
   ```

### Making Predictions

```python
from src.models.hybrid import HybridModel
import torch
from torchvision import transforms
from PIL import Image

# Initialize and load model
model = HybridModel(num_classes=38, backbone_name='efficientnet_b3')
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('path_to_image.jpg')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
```

### Exporting Models

#### To ONNX
```bash
python scripts/export_model.py \
    --checkpoint models/best_model.pth \
    --output models/plant_disease_model.onnx \
    --img_size 256
```

#### To TensorFlow Lite
```bash
python scripts/export_model.py \
    --checkpoint models/best_model.pth \
    --output models/plant_disease_model.tflite \
    --img_size 256 \
    --target tflite
```.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/VIKAS9793/KrishiSahayak.git
   cd KrishiSahayak
   ```

2. **Set up a virtual environment** (recommended)
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the package**
   ```bash
   # Install in development mode with all optional dependencies
   pip install -e ".[dev,train,data]"
   
   # Or for production (minimal dependencies)
   # pip install .
   # Install everything (dev + train + deploy)
   pip install -e ".[all]"
   ```

4. **Verify installation**
   ```bash
   python -c "import krishisahayak; print('KrishiSahayak version:', krishisahayak.__version__)"
   ```

## 🏗️ Project Structure

```
KrishiSahayak/
├── README.md                # You are here!
├── LICENSE                  # Project license
├── requirements.txt         # Python dependencies
│
├── assets/                  # Banners and static images for UI
├── configs/                 # Configuration files (e.g., default.yaml)
├── data/                    # For storing raw and processed datasets
├── docs/                    # All project documentation
├── models/                  # For storing trained model checkpoints (teacher/student)
├── reports/                 # For data validation and model evaluation reports
├── scripts/                 # Helper scripts for data processing and model exporting
└── src/                     # All source code for the project

├── configs/                    # Configuration files
│   └── train_config.yaml      # Training configuration
├── data/                      # Dataset directory
├── scripts/                   # Utility scripts
│   ├── data/
│   │   └── generate_synthetic_ms.py  # Generate synthetic MS data
│   └── train.py               # Main training script
├── src/                       # Source code
│   ├── data/
│   │   └── dataset.py        # Data loading and preprocessing
│   ├── models/
│   │   ├── base.py          # Base model class
│   │   └── hybrid.py         # Hybrid RGB-MS model
│   └── web/
│       └── app.py           # Web interface
└── README.md                  # This file
```

## 🚀 Quick Start

### Training

1. Prepare your dataset in the following structure:
   ```
   data/
   ├── metadata.csv      # CSV with columns: image_path, label, split
   ├── images/           # RGB images
   └── multispectral/    # Optional: Multispectral images
   ```

2. Update the configuration in `configs/train_config.yaml`

3. Start training:
   ```bash
   python scripts/train.py --config configs/train_config.yaml
   ```

### Web Interface

Launch the web interface:
```bash
python src/web/app.py
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please open an issue or contact the maintainers.
- CPU: Minimum 4 cores (8+ recommended)
- RAM: Minimum 8GB (16GB+ recommended)
- GPU: Optional but highly recommended for faster training (NVIDIA GPU with CUDA 11.8 support).
Software:
- Python 3.9 or 3.10
- pip (Python package installer)

2. Installation & Setup
# 1. Clone the repository
git clone [https://github.com/VIKAS9793/KrishiSahayak.git](https://github.com/VIKAS9793/KrishiSahayak.git)
cd KrishiSahayak

# 2. Install Python dependencies
pip install -r requirements.txt

3. Data Preparation
The model expects the dataset to be organized in a flat folder structure by class, with all images referenced in a metadata CSV. Each subfolder should contain images for a single class. Example:

data/plantvillage/
├── soybean_healthy/
│   ├── image_001.jpg
│   └── ...
├── tomato_late_blight/
│   ├── image_001.jpg
│   └── ...
└── ...

A metadata CSV (`data/metadata.csv`) is used to track all images, their class labels, and their source. The CSV is generated and cleaned using the main validation script.

4. Data Validation & Cleaning
Use the following script to validate your data, check for missing/corrupt files, and ensure your metadata CSV is in sync:

python scripts/data/check_datamodule.py

This script will:
- Validate your environment and dependencies
- Analyze metadata for missing/duplicate values and class balance
- Perform a file integrity check (removes missing/corrupt files from the CSV)
- Test DataModule and DataLoader

5. Training the Model
You can start training with default parameters or provide your own.

# Basic training with default settings from the config file
python src/train.py

# (Example) Override default parameters for a custom run
python src/train.py \
    --model efficientnet_b3 \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001

6. Monitoring Training
You can monitor training progress, including loss and accuracy metrics, using TensorBoard.

# Launch TensorBoard to view logs from the project root directory
tensorboard --logdir=logs/

🏗️ Project Architecture
KrishiSahayak uses a Teacher-Student distillation strategy to achieve both high accuracy and high performance.

- Teacher Model (EfficientNetV2): A large, highly accurate model used as the source of knowledge. It is not intended for deployment due to its size.
- Student Model (MobileNetV3-Large): A lightweight, efficient model specifically designed for fast inference on mobile and edge devices. It is trained to mimic the behavior of the teacher model, preserving high accuracy while being resource-efficient.
- Training Pipeline: Involves mixed precision, learning rate scheduling (Cosine Annealing), and handling class imbalance with weighted sampling to ensure robust training.

For a more detailed technical breakdown, see the docs/ARCHITECTURE.md file.

💾 Data Sources & Rationale
- **Current workflow uses PlantVillage as the primary dataset** for training and validation, due to its size and quality.
- PlantDoc is optional and can be added for future robustness and real-world generalization.
- Using both datasets (when enabled) allows the model to learn from clean examples while also being resilient to the variability of real-world conditions.

⚠️ Class Imbalance
- The dataset exhibits severe class imbalance (up to 36:1 ratio).
- It is strongly recommended to use class weighting, oversampling, or advanced loss functions (e.g., focal loss) during training to mitigate this issue.

📚 Documentation
- docs/ARCHITECTURE.md – System design and model architecture
- docs/QUICKSTART.md – Quick start guide
- docs/journey.md – Project story and purpose

📁 Project Structure
KrishiSahayak/
├── README.md                # You are here!
├── LICENSE                  # Project license
├── requirements.txt         # Python dependencies
│
├── assets/                  # Banners and static images for UI
├── configs/                 # Configuration files (e.g., default.yaml)
├── data/                    # For storing raw and processed datasets
├── docs/                    # All project documentation
├── models/                  # For storing trained model checkpoints (teacher/student)
├── reports/                 # For data validation and model evaluation reports
├── scripts/                 # Helper scripts for data processing and model exporting
└── src/                     # All source code for the project

⚖️ License
This project is licensed under the MIT License. See the LICENSE file for details.
