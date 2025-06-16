# Quick Start Guide

This guide will help you set up the development environment and get started with the KrishiRakshak project.

## Prerequisites

- Python 3.10 (recommended) or 3.9
- pip (Python package installer)
- CUDA (optional, for GPU acceleration)

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
