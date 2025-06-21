"""
Enhanced Model Export Utility for KrishiSahayak.

This script exports a trained PyTorch Lightning checkpoint to deployment-ready
formats like ONNX and TensorFlow Lite, with support for quantization and
comprehensive validation.
"""
import argparse
import logging
import yaml
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import onnx
import onnx_tf.backend
import tensorflow as tf
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

# In a real setup, this is handled by installing the package with `pip install -e .`
from src.models.hybrid import HybridModel
from src.data.dataset import PlantDiseaseDataset
from albumentations import Resize, Normalize, Compose
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExportError(Exception):
    """Custom exception for export-related errors."""
    pass

class HybridModelForExport(nn.Module):
    """
    A wrapper for the HybridModel that provides a simple, tensor-based
    forward signature suitable for ONNX tracing.
    """
    def __init__(self, model: HybridModel):
        super().__init__()
        self.model = model
        self.use_ms = model.use_ms

    def forward(self, image_input: torch.Tensor, ms_input: Optional[torch.Tensor] = None):
        """Forward pass with optional multispectral input."""
        batch = {'image': image_input}
        if self.use_ms and ms_input is not None:
            batch['ms_data'] = ms_input
        return self.model(batch)

class RepresentativeDataset:
    """
    A generator that provides real samples from a DataLoader for TFLite int8 quantization.
    Implements proper iteration and error handling.
    """
    def __init__(self, dataloader: DataLoader, use_ms: bool, max_samples: int = 100):
        self.dataloader = dataloader
        self.use_ms = use_ms
        self.max_samples = min(max_samples, len(dataloader))
        self.sample_count = 0

    def __call__(self):
        """Generator function for representative samples."""
        logger.info(f"Generating {self.max_samples} representative samples for quantization...")
        
        for batch_idx, batch in enumerate(self.dataloader):
            if self.sample_count >= self.max_samples:
                break
                
            try:
                image_input = batch['image'].numpy()
                if self.use_ms and 'ms_data' in batch:
                    ms_input = batch['ms_data'].numpy()
                    yield [image_input, ms_input]
                else:
                    yield [image_input]
                
                self.sample_count += 1
                
                if (self.sample_count % 20) == 0:
                    logger.info(f"Generated {self.sample_count}/{self.max_samples} samples")
                    
            except Exception as e:
                logger.warning(f"Skipping batch {batch_idx} due to error: {e}")
                continue

def validate_onnx_model(onnx_path: Path, original_model: HybridModelForExport, 
                       config: Dict[str, Any]) -> bool:
    """Validate ONNX model by comparing outputs with original PyTorch model."""
    try:
        import onnxruntime as ort
        logger.info("Validating ONNX model...")
        
        # Load ONNX model
        ort_session = ort.InferenceSession(str(onnx_path))
        
        # Create test inputs
        img_size = config['data']['image_size']
        test_rgb = torch.randn(1, 3, *img_size)
        
        # Get PyTorch output
        original_model.eval()
        with torch.no_grad():
            if original_model.use_ms:
                test_ms = torch.randn(1, config['model']['ms_channels'], *img_size)
                pytorch_output = original_model(test_rgb, test_ms).numpy()
                onnx_inputs = {
                    'image_input': test_rgb.numpy(),
                    'ms_input': test_ms.numpy()
                }
            else:
                pytorch_output = original_model(test_rgb).numpy()
                onnx_inputs = {'image_input': test_rgb.numpy()}
        
        # Get ONNX output
        onnx_output = ort_session.run(None, onnx_inputs)[0]
        
        # Compare outputs
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        logger.info(f"Maximum difference between PyTorch and ONNX outputs: {max_diff}")
        
        if max_diff < 1e-5:
            logger.info("✅ ONNX model validation successful")
            return True
        else:
            logger.warning(f"⚠️ ONNX model validation failed - difference too large: {max_diff}")
            return False
            
    except ImportError:
        logger.warning("ONNXRuntime not available - skipping validation")
        return True
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return False

def get_model_size(file_path: Path) -> str:
    """Get human-readable file size."""
    size_bytes = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def export_to_onnx(model_wrapper: HybridModelForExport, output_path: Path, 
                   config: Dict[str, Any]) -> bool:
    """Exports the model wrapper to ONNX format with validation."""
    logger.info(f"Starting ONNX export to {output_path}...")
    start_time = time.time()
    
    try:
        model_wrapper.eval()
        
        img_size = config['data']['image_size']
        deploy_cfg = config['deployment']
        
        # Create dummy inputs based on whether the model is hybrid
        dummy_rgb = torch.randn(1, 3, *img_size, requires_grad=True)
        inputs = (dummy_rgb,)
        input_names = ['image_input']
        dynamic_axes = {'image_input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
        if model_wrapper.use_ms:
            dummy_ms = torch.randn(1, config['model']['ms_channels'], *img_size)
            inputs = (dummy_rgb, dummy_ms)
            input_names.append('ms_input')
            dynamic_axes['ms_input'] = {0: 'batch_size'}

        # Export to ONNX
        torch.onnx.export(
            model_wrapper,
            inputs,
            str(output_path),
            export_params=True,
            opset_version=deploy_cfg['onnx']['opset_version'],
            do_constant_folding=True,
            input_names=input_names,
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        export_time = time.time() - start_time
        model_size = get_model_size(output_path)
        logger.info(f"✅ ONNX export successful in {export_time:.2f}s - Size: {model_size}")
        
        # Validate the exported model
        is_valid = validate_onnx_model(output_path, model_wrapper, config)
        return is_valid
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise ExportError(f"ONNX export failed: {e}")

def export_to_tflite(onnx_path: Path, output_dir: Path, model: HybridModel, 
                     config: Dict[str, Any]) -> List[Path]:
    """Converts an ONNX model to TensorFlow Lite, with quantization options."""
    exported_files = []
    
    try:
        logger.info("Starting ONNX to TensorFlow conversion...")
        start_time = time.time()
        
        onnx_model = onnx.load(onnx_path)
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        tf_model_path = output_dir / "tf_model"
        tf_rep.export_graph(str(tf_model_path))
        
        conversion_time = time.time() - start_time
        logger.info(f"✅ ONNX to TensorFlow conversion successful in {conversion_time:.2f}s")

        logger.info("Starting TensorFlow to TFLite conversion...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
        
        # Base model (FP32)
        base_model = converter.convert()
        base_path = output_dir / f"{onnx_path.stem}_fp32.tflite"
        base_path.write_bytes(base_model)
        exported_files.append(base_path)
        logger.info(f"✅ Base FP32 TFLite model saved - Size: {get_model_size(base_path)}")
        
        # Quantization options
        quant_cfg = config['deployment']['tflite']['quantization']
        if quant_cfg['enabled']:
            # Float16 Quantization
            if quant_cfg['mode'] in ['float16', 'both']:
                logger.info("Applying float16 quantization...")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                
                fp16_model = converter.convert()
                fp16_path = output_dir / f"{onnx_path.stem}_fp16.tflite"
                fp16_path.write_bytes(fp16_model)
                exported_files.append(fp16_path)
                logger.info(f"✅ Float16 TFLite model saved - Size: {get_model_size(fp16_path)}")
            
            # INT8 Quantization
            if quant_cfg['mode'] in ['int8', 'both']:
                logger.info("Applying int8 quantization with representative dataset...")
                converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # Create representative dataset
                val_transform = Compose([
                    Resize(*config['data']['image_size']), 
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
                
                val_dataset = PlantDiseaseDataset(
                    config['data']['csv_path'], 
                    config['data']['rgb_dir'], 
                    'val', 
                    val_transform, 
                    model.use_ms, 
                    config['data'].get('ms_dir')
                )
                
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
                rep_dataset = RepresentativeDataset(val_loader, model.use_ms, max_samples=100)
                
                converter.representative_dataset = rep_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                
                int8_model = converter.convert()
                int8_path = output_dir / f"{onnx_path.stem}_int8.tflite"
                int8_path.write_bytes(int8_model)
                exported_files.append(int8_path)
                logger.info(f"✅ INT8 TFLite model saved - Size: {get_model_size(int8_path)}")
        
        return exported_files
        
    except Exception as e:
        logger.error(f"TFLite export failed: {e}")
        raise ExportError(f"TFLite export failed: {e}")

def print_export_summary(exported_files: List[Path], original_checkpoint: Path):
    """Print a summary of exported models."""
    logger.info("\n" + "="*60)
    logger.info("EXPORT SUMMARY")
    logger.info("="*60)
    
    original_size = get_model_size(original_checkpoint)
    logger.info(f"Original checkpoint: {original_checkpoint.name} ({original_size})")
    logger.info(f"Exported {len(exported_files)} model(s):")
    
    for file_path in exported_files:
        size = get_model_size(file_path)
        logger.info(f"  • {file_path.name} ({size})")
    
    logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Export a trained model to ONNX or TFLite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_model.py --checkpoint model.ckpt --config config.yaml --format onnx
  python export_model.py --checkpoint model.ckpt --config config.yaml --format tflite
  python export_model.py --checkpoint model.ckpt --config config.yaml --format tflite --validate
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to the model checkpoint (.ckpt) file.')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to the project configuration YAML file.')
    parser.add_argument('--format', type=str, choices=['onnx', 'tflite', 'both'], 
                       required=True, help='Export format(s)')
    parser.add_argument('--output-dir', type=str, 
                       help='Directory to save the exported model.')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate exported models')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Setup output directory
        output_dir = Path(args.output_dir or config['project']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify checkpoint exists
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        
        # Load model
        logger.info("Loading model from checkpoint...")
        trained_model = HybridModel.load_from_checkpoint(args.checkpoint)
        model_wrapper = HybridModelForExport(trained_model)
        
        # Export models
        model_name = checkpoint_path.stem
        onnx_path = output_dir / f"{model_name}.onnx"
        exported_files = []
        
        if args.format in ['onnx', 'both']:
            success = export_to_onnx(model_wrapper, onnx_path, config)
            if success:
                exported_files.append(onnx_path)
            elif not args.format == 'both':
                raise ExportError("ONNX export failed validation")
        
        if args.format in ['tflite', 'both']:
            if not onnx_path.exists():
                logger.info("ONNX model not found, creating it first...")
                success = export_to_onnx(model_wrapper, onnx_path, config)
                if success:
                    exported_files.append(onnx_path)
            
            tflite_files = export_to_tflite(onnx_path, output_dir, trained_model, config)
            exported_files.extend(tflite_files)
        
        # Print summary
        print_export_summary(exported_files, checkpoint_path)
        logger.info("🎉 Export process completed successfully!")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())