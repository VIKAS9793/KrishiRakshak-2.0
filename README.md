# KrishiRakshak: Plant Disease Classification

An end-to-end deep learning pipeline for plant disease classification using PyTorch Lightning and modern MLOps practices.

## Architecture Overview

For detailed technical architecture and model specifications, please refer to the [Architecture Documentation](docs/ARCHITECTURE.md).

## Project Structure

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
