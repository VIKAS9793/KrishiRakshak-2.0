"""
Data Validation Framework for Plant Disease Classification

This script provides a scalable and extensible framework for dataset validation.
It uses a pluggable "Check Runner" pattern and parallel processing to handle
large datasets efficiently, reflecting production standards for MLOps tooling.
"""
import abc
import json
import logging
import argparse
import multiprocessing
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm

# --- Configuration ---
# In a real MAANG environment, this would come from a YAML/JSON config file.
IMAGE_QUALITY_CONFIG = {
    "sample_size": 1000,
    "min_dimension": 224,
    "aspect_ratio_range": (0.5, 2.0),
}
LABEL_ANALYSIS_CONFIG = {"min_samples_threshold": 10}
SPLIT_ANALYSIS_CONFIG = {
    "expected_splits": ["train", "val", "test"],
    "min_split_ratio": 0.05,  # Each split should have at least 5% of data
    "max_split_ratio": 0.85   # No split should have more than 85% of data
}
MS_DATA_CONFIG = {
    "min_bands": 3,           # Minimum number of spectral bands required
    "max_bands": 8,           # Maximum number of spectral bands expected
    "band_names": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08"],  # Common MS band names
    "min_pixel_value": 0,     # Minimum valid pixel value
    "max_pixel_value": 1.0,   # Maximum valid pixel value (assuming normalized)
    "check_band_consistency": True,  # Check if all images have same number of bands
    "check_band_order": True,        # Check if band order is consistent
    "check_nodata": True,            # Check for no-data values
    "check_metadata_consistency": True  # Check if MS metadata matches RGB metadata
}

# Configure logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Abstract Base Class for all Checks ---

class ValidationCheck(abc.ABC):
    """Abstract base class for a single validation check."""
    def __init__(self, validator: 'DataValidator'):
        self.validator = validator

    @abc.abstractmethod
    def run(self) -> Dict[str, Any]:
        """Runs the validation check and returns a results dictionary."""
        raise NotImplementedError

# --- Concrete Check Implementations ---

class StructureCheck(ValidationCheck):
    """Validates metadata structure, file existence, and duplicates."""
    def run(self) -> Dict[str, Any]:
        logger.info("Running: Structure Validation")
        results: Dict[str, Any] = {'file_checks': {}, 'warnings': [], 'errors': []}
        df = self.validator.metadata.copy()

        required_cols = ['image_path', 'label', 'split']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            msg = f"Missing required columns in metadata: {missing_cols}"
            logger.error(msg)
            results['errors'].append(msg)
            return results

        df['exists'] = df['image_path'].apply(lambda x: (self.validator.data_dir / x).exists())
        missing_files = df[~df['exists']]
        if not missing_files.empty:
            msg = f"Found {len(missing_files)} missing file paths in metadata."
            logger.warning(msg)
            results['warnings'].append(msg)
            missing_files[['image_path', 'label', 'split']].to_csv(
                self.validator.output_dir / 'missing_files.csv', index=False
            )

        duplicates = df.duplicated(subset=['image_path'], keep=False)
        if duplicates.any():
            msg = f"Found {duplicates.sum()} duplicate file paths in metadata."
            logger.warning(msg)
            results['warnings'].append(msg)
            df[duplicates].to_csv(self.validator.output_dir / 'duplicate_files.csv', index=False)

        results['file_checks'] = {
            'total_files': len(df),
            'missing_files_count': len(missing_files),
            'duplicate_paths_count': int(duplicates.sum())
        }

        if self.validator.fix:
            logger.info("--fix enabled: Removing missing file entries from metadata.")
            cleaned_df = self.validator.metadata[df['exists']].drop_duplicates(subset=['image_path'])
            self.validator.metadata = cleaned_df # Update the main dataframe
            logger.info(f"Metadata cleaned. New size: {len(self.validator.metadata)}")

        return results

def _validate_single_image(args_tuple):
    """Helper function for parallel image processing."""
    img_path_str, data_dir_str, allow_truncated = args_tuple
    img_path = Path(data_dir_str) / img_path_str
    
    # Configure truncated image loading per-process
    ImageFile.LOAD_TRUNCATED_IMAGES = allow_truncated
    
    try:
        with Image.open(img_path) as img:
            img.load() # Force loading the image data to catch truncation errors if not allowed
            img_format = img.format or 'unknown'
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            width, height = img.size
            aspect_ratio = round(width / height, 2)
            
            is_small = min(width, height) < IMAGE_QUALITY_CONFIG['min_dimension']
            is_unusual_ar = not (IMAGE_QUALITY_CONFIG['aspect_ratio_range'][0] <= aspect_ratio <= IMAGE_QUALITY_CONFIG['aspect_ratio_range'][1])
            
            return {
                'path': img_path_str,
                'status': 'ok',
                'stats': {
                    'height': height, 'width': width, 'aspect_ratio': aspect_ratio,
                    'channels': len(img.getbands()), 'format': img_format
                },
                'is_small': is_small,
                'is_unusual_ar': is_unusual_ar
            }
    except (IOError, UnidentifiedImageError, SyntaxError) as e:
        return {'path': img_path_str, 'status': 'corrupt', 'error': str(e)}

class ImageQualityCheck(ValidationCheck):
    """Validates image quality in parallel."""
    def run(self) -> Dict[str, Any]:
        logger.info("Running: Image Quality Validation (in parallel)")
        results: Dict[str, Any] = {'image_quality': {}, 'warnings': []}
        df = self.validator.metadata

        sample = df if len(df) <= IMAGE_QUALITY_CONFIG['sample_size'] else df.sample(IMAGE_QUALITY_CONFIG['sample_size'])
        image_paths = sample['image_path'].tolist()
        
        # Prepare arguments for the parallel pool
        args_list = [(path, str(self.validator.data_dir), self.validator.allow_truncated) for path in image_paths]

        # Use multiprocessing to parallelize the IO-bound task
        pool_size = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using {pool_size} processes for image validation.")
        
        processed_results = []
        with multiprocessing.Pool(processes=pool_size) as pool:
            with tqdm(total=len(image_paths), desc="Validating images") as pbar:
                for result in pool.imap_unordered(_validate_single_image, args_list):
                    processed_results.append(result)
                    pbar.update()

        # Aggregate results
        corrupt = [r for r in processed_results if r['status'] == 'corrupt']
        ok = [r for r in processed_results if r['status'] == 'ok']
        
        if corrupt:
            msg = f"Found {len(corrupt)} corrupt images."
            logger.warning(msg)
            results['warnings'].append(msg)
            (self.validator.output_dir / 'corrupt_images.json').write_text(json.dumps(corrupt, indent=2))

        # Compile stats from valid images
        stats = {
            'heights': [r['stats']['height'] for r in ok],
            'widths': [r['stats']['width'] for r in ok],
            'aspect_ratios': [r['stats']['aspect_ratio'] for r in ok],
            'formats': pd.Series([r['stats']['format'] for r in ok]).value_counts().to_dict()
        }

        results['image_quality'] = {
            'processed_count': len(processed_results),
            'corrupt_count': len(corrupt),
            'small_image_count': sum(1 for r in ok if r['is_small']),
            'unusual_ar_count': sum(1 for r in ok if r['is_unusual_ar']),
            'stats': stats
        }
        return results

class LabelCheck(ValidationCheck):
    """Validates label distribution and consistency."""
    def run(self) -> Dict[str, Any]:
        logger.info("Running: Label Distribution Validation")
        results: Dict[str, Any] = {'label_analysis': {}, 'warnings': [], 'errors': []}
        df = self.validator.metadata

        if 'label' not in df.columns:
            msg = "Label column not found in metadata."
            logger.error(msg)
            results['errors'].append(msg)
            return results

        # Analyze label distribution
        label_counts = df['label'].value_counts()
        total_samples = len(df)
        
        # Check for classes with too few samples
        min_threshold = LABEL_ANALYSIS_CONFIG['min_samples_threshold']
        underrepresented = label_counts[label_counts < min_threshold]
        
        if not underrepresented.empty:
            msg = f"Found {len(underrepresented)} classes with fewer than {min_threshold} samples."
            logger.warning(msg)
            results['warnings'].append(msg)
            underrepresented.to_csv(self.validator.output_dir / 'underrepresented_classes.csv', 
                                  header=['count'])

        # Check for label consistency (no leading/trailing spaces, case issues)
        problematic_labels = []
        for label in df['label'].unique():
            if str(label) != str(label).strip():
                problematic_labels.append(label)
        
        if problematic_labels:
            msg = f"Found {len(problematic_labels)} labels with whitespace issues."
            logger.warning(msg)
            results['warnings'].append(msg)
            pd.DataFrame({'problematic_labels': problematic_labels}).to_csv(
                self.validator.output_dir / 'problematic_labels.csv', index=False
            )

        # Calculate class imbalance metrics
        label_proportions = label_counts / total_samples
        max_proportion = label_proportions.max()
        min_proportion = label_proportions.min()
        imbalance_ratio = max_proportion / min_proportion if min_proportion > 0 else float('inf')

        results['label_analysis'] = {
            'total_classes': len(label_counts),
            'total_samples': total_samples,
            'class_distribution': label_counts.to_dict(),
            'class_proportions': label_proportions.to_dict(),
            'underrepresented_classes_count': len(underrepresented),
            'imbalance_ratio': float(imbalance_ratio),
            'problematic_labels_count': len(problematic_labels)
        }

        # Auto-fix label issues if requested
        if self.validator.fix and problematic_labels:
            logger.info("--fix enabled: Cleaning label whitespace issues.")
            self.validator.metadata['label'] = self.validator.metadata['label'].astype(str).str.strip()
            logger.info("Label whitespace issues fixed.")

        return results

class SplitCheck(ValidationCheck):
    """Validates data split distribution and consistency."""
    def run(self) -> Dict[str, Any]:
        logger.info("Running: Data Split Validation")
        results: Dict[str, Any] = {'split_analysis': {}, 'warnings': [], 'errors': []}
        df = self.validator.metadata

        if 'split' not in df.columns:
            msg = "Split column not found in metadata."
            logger.error(msg)
            results['errors'].append(msg)
            return results

        # Analyze split distribution
        split_counts = df['split'].value_counts()
        total_samples = len(df)
        split_proportions = split_counts / total_samples
        
        # Check for expected splits
        expected_splits = set(SPLIT_ANALYSIS_CONFIG['expected_splits'])
        actual_splits = set(split_counts.index)
        missing_splits = expected_splits - actual_splits
        unexpected_splits = actual_splits - expected_splits

        if missing_splits:
            msg = f"Missing expected splits: {missing_splits}"
            logger.warning(msg)
            results['warnings'].append(msg)

        if unexpected_splits:
            msg = f"Found unexpected splits: {unexpected_splits}"
            logger.warning(msg)
            results['warnings'].append(msg)

        # Check split ratio bounds
        min_ratio = SPLIT_ANALYSIS_CONFIG['min_split_ratio']
        max_ratio = SPLIT_ANALYSIS_CONFIG['max_split_ratio']
        
        problematic_splits = []
        for split_name, proportion in split_proportions.items():
            if proportion < min_ratio:
                problematic_splits.append(f"{split_name}: {proportion:.3f} (too small)")
            elif proportion > max_ratio:
                problematic_splits.append(f"{split_name}: {proportion:.3f} (too large)")

        if problematic_splits:
            msg = f"Split ratio issues: {', '.join(problematic_splits)}"
            logger.warning(msg)
            results['warnings'].append(msg)

        # Check class distribution across splits
        cross_tab = pd.crosstab(df['label'], df['split'], normalize='index')
        split_consistency_issues = []
        
        for class_name in cross_tab.index:
            class_splits = cross_tab.loc[class_name]
            # Check if any split has 0 samples for this class
            missing_in_splits = class_splits[class_splits == 0].index.tolist()
            if missing_in_splits and len(missing_in_splits) < len(class_splits):
                split_consistency_issues.append(f"Class '{class_name}' missing in splits: {missing_in_splits}")

        if split_consistency_issues:
            msg = f"Found {len(split_consistency_issues)} class-split consistency issues."
            logger.warning(msg)
            results['warnings'].append(msg)
            pd.DataFrame({'issues': split_consistency_issues}).to_csv(
                self.validator.output_dir / 'split_consistency_issues.csv', index=False
            )

        results['split_analysis'] = {
            'total_samples': total_samples,
            'split_distribution': split_counts.to_dict(),
            'split_proportions': split_proportions.to_dict(),
            'missing_expected_splits': list(missing_splits),
            'unexpected_splits': list(unexpected_splits),
            'problematic_ratios_count': len(problematic_splits),
            'class_split_consistency_issues': len(split_consistency_issues)
        }

        # Save detailed cross-tabulation
        cross_tab.to_csv(self.validator.output_dir / 'class_split_crosstab.csv')

        return results

class DataLeakageCheck(ValidationCheck):
    """Checks for potential data leakage between splits."""
    def run(self) -> Dict[str, Any]:
        logger.info("Running: Data Leakage Detection")
        results: Dict[str, Any] = {'leakage_analysis': {}, 'warnings': [], 'errors': []}
        df = self.validator.metadata

        # Check for identical image paths across different splits
        if 'split' not in df.columns:
            return results  # Skip if no split column

        # Group by image path and check if any path appears in multiple splits
        path_splits = df.groupby('image_path')['split'].apply(set)
        multi_split_paths = path_splits[path_splits.apply(len) > 1]
        
        if not multi_split_paths.empty:
            msg = f"Found {len(multi_split_paths)} images appearing in multiple splits."
            logger.error(msg)
            results['errors'].append(msg)
            
            leakage_details = []
            for path, splits in multi_split_paths.items():
                leakage_details.append({'image_path': path, 'splits': list(splits)})
            
            pd.DataFrame(leakage_details).to_csv(
                self.validator.output_dir / 'data_leakage_detected.csv', index=False
            )

        results['leakage_analysis'] = {
            'images_in_multiple_splits': len(multi_split_paths),
            'total_unique_images': len(path_splits)
        }

        return results

# --- The Main Validator Class (Check Runner) ---

class DataValidator:
    """A scalable and extensible data validation framework."""
    def __init__(self, data_dir: str, metadata_path: str, output_dir: str, fix: bool, allow_truncated: bool):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fix = fix
        self.allow_truncated = allow_truncated
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.metadata = pd.read_csv(metadata_path)
            logger.info(f"Successfully loaded metadata with {len(self.metadata)} records.")
        except FileNotFoundError:
            logger.error(f"Metadata file not found at: {metadata_path}")
            raise

        self.results: Dict[str, Any] = {'summary': {}, 'checks': {}}
        
        # The pluggable check system
        self.checks: List[ValidationCheck] = [
            StructureCheck(self),
            ImageQualityCheck(self),
            LabelCheck(self),
            SplitCheck(self),
            DataLeakageCheck(self),
            MSDataQualityCheck(self),
        ]
        
    def run_all_checks(self):
        """Execute all registered validation checks."""
        for check in self.checks:
            try:
                check_result = check.run()
                # Merge results, appending warnings/errors
                for key, value in check_result.items():
                    if key in ['warnings', 'errors']:
                        self.results.setdefault(key, []).extend(value)
                    else:
                        self.results['checks'][key] = value
            except Exception as e:
                logger.error(f"Check {type(check).__name__} failed: {e}", exc_info=True)
                self.results.setdefault('errors', []).append(f"FATAL: {type(check).__name__} failed.")

    def generate_report(self):
        """Generate and save the final validation report and visualizations."""
        logger.info("Generating final validation report...")
        self.results['summary'] = {
            'total_samples_final': len(self.metadata),
            'warnings_count': len(self.results.get('warnings', [])),
            'errors_count': len(self.results.get('errors', [])),
            'validation_passed': len(self.results.get('errors', [])) == 0,
        }
        report_path = self.output_dir / 'validation_report.json'
        
        # A helper to handle non-serializable numpy types
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)

        report_path.write_text(json.dumps(self.results, indent=2, cls=NpEncoder))
        logger.info(f"Full validation report saved to {report_path}")
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Print summary to console
        self._print_summary()

    def _generate_visualizations(self):
        """Generate visualizations for the validation report."""
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Image dimension visualization
        if 'image_quality' in self.results['checks']:
            stats = self.results['checks']['image_quality']['stats']
            if stats.get('heights'):
                plt.figure(figsize=(12, 5))
                
                # Subplot 1: Dimension scatter
                plt.subplot(1, 2, 1)
                plt.scatter(stats['widths'], stats['heights'], alpha=0.6, s=20)
                plt.xlabel('Width (px)')
                plt.ylabel('Height (px)')
                plt.title('Image Dimensions Distribution')
                plt.grid(True, alpha=0.3)
                
                # Subplot 2: Aspect ratio histogram
                plt.subplot(1, 2, 2)
                plt.hist(stats['aspect_ratios'], bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('Aspect Ratio')
                plt.ylabel('Frequency')
                plt.title('Aspect Ratio Distribution')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(viz_dir / 'image_dimensions.png', dpi=150, bbox_inches='tight')
                plt.close()

        # Label distribution visualization
        if 'label_analysis' in self.results['checks']:
            label_data = self.results['checks']['label_analysis']
            if 'class_distribution' in label_data:
                plt.figure(figsize=(12, 6))
                
                class_counts = label_data['class_distribution']
                classes = list(class_counts.keys())
                counts = list(class_counts.values())
                
                plt.bar(classes, counts)
                plt.xlabel('Class Labels')
                plt.ylabel('Sample Count')
                plt.title('Class Distribution')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
                plt.close()

        # Split distribution visualization
        if 'split_analysis' in self.results['checks']:
            split_data = self.results['checks']['split_analysis']
            if 'split_distribution' in split_data:
                plt.figure(figsize=(8, 8))
                
                split_counts = split_data['split_distribution']
                labels = list(split_counts.keys())
                sizes = list(split_counts.values())
                
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                plt.title('Data Split Distribution')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(viz_dir / 'split_distribution.png', dpi=150, bbox_inches='tight')
                plt.close()

    def _print_summary(self):
        """Print a concise summary to the console."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        summary = self.results['summary']
        print(f"Total Samples: {summary['total_samples_final']}")
        print(f"Warnings: {summary['warnings_count']}")
        print(f"Errors: {summary['errors_count']}")
        print(f"Validation Status: {'PASSED' if summary['validation_passed'] else 'FAILED'}")
        
        if self.results.get('errors'):
            print("\nCRITICAL ERRORS:")
            for error in self.results['errors']:
                print(f"  ❌ {error}")
        
        if self.results.get('warnings'):
            print(f"\nWARNINGS ({len(self.results['warnings'])}):")
            for warning in self.results['warnings'][:5]:  # Show first 5 warnings
                print(f"  ⚠️  {warning}")
            if len(self.results['warnings']) > 5:
                print(f"  ... and {len(self.results['warnings']) - 5} more warnings")
        
        print(f"\nDetailed report saved to: {self.output_dir / 'validation_report.json'}")
        print("="*60)

def parse_args():
    parser = argparse.ArgumentParser(description='A Scalable Data Validation Framework for CV Datasets.')
    parser.add_argument('--data-dir', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--metadata', type=str, required=True, help='Path to the metadata CSV file.')
    parser.add_argument('--output-dir', type=str, default='reports', help='Directory for saving reports.')
    parser.add_argument('--fix', action='store_true', help='Enable auto-fixing of issues like missing files.')
    parser.add_argument('--allow-truncated', action='store_true', help='Allow loading of truncated images.')
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        validator = DataValidator(
            data_dir=args.data_dir,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            fix=args.fix,
            allow_truncated=args.allow_truncated
        )
        validator.run_all_checks()
        validator.generate_report()
        logger.info("Validation completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred during validation: {str(e)}", exc_info=True)
        sys.exit(1)

class MSDataQualityCheck(ValidationCheck):
    """Validates multispectral data quality and consistency."""
    
    def __init__(self, validator: 'DataValidator'):
        super().__init__(validator)
        self.config = MS_DATA_CONFIG
        
    def _load_ms_image(self, ms_path: Path) -> Optional[np.ndarray]:
        """Load MS image with error handling."""
        try:
            # For demonstration - adjust based on actual MS data format
            # This might need to be updated for your specific MS data format
            ms_img = np.load(ms_path) if ms_path.suffix == '.npy' else np.array(Image.open(ms_path))
            return ms_img
        except Exception as e:
            logger.warning(f"Error loading MS image at {ms_path}: {e}")
            return None
    
    def _validate_ms_image(self, ms_path: Path) -> Dict[str, Any]:
        """Validate a single MS image."""
        ms_img = self._load_ms_image(ms_path)
        if ms_img is None:
            return {"valid": False, "error": "Failed to load MS image"}
            
        results = {
            "valid": True,
            "shape": ms_img.shape,
            "dtype": str(ms_img.dtype),
            "min_val": float(ms_img.min()),
            "max_val": float(ms_img.max()),
            "has_nan": bool(np.isnan(ms_img).any()),
            "has_inf": bool(np.isinf(ms_img).any()),
            "band_count": ms_img.shape[2] if len(ms_img.shape) > 2 else 1
        }
        
        # Check band count
        if not (self.config["min_bands"] <= results["band_count"] <= self.config["max_bands"]):
            results["valid"] = False
            results["error"] = f"Invalid band count: {results['band_count']}"
            
        # Check pixel value range
        if (results["min_val"] < self.config["min_pixel_value"] or 
            results["max_val"] > self.config["max_pixel_value"]):
            results["valid"] = False
            results["error"] = f"Pixel values out of range: [{results['min_val']}, {results['max_val']}]"
            
        return results
    
    def run(self) -> Dict[str, Any]:
        """Run MS data quality checks."""
        logger.info("Running: MS Data Quality Validation")
        results = {
            'ms_checks': {'valid': [], 'invalid': []},
            'band_distribution': {},
            'warnings': [],
            'errors': []
        }
        
        if 'ms_path' not in self.validator.metadata.columns:
            msg = "No 'ms_path' column found in metadata. Skipping MS data validation."
            logger.warning(msg)
            results['warnings'].append(msg)
            return results
            
        # Filter out rows with missing MS paths
        ms_df = self.validator.metadata.dropna(subset=['ms_path']).copy()
        if ms_df.empty:
            msg = "No valid MS paths found in metadata."
            logger.warning(msg)
            results['warnings'].append(msg)
            return results
            
        # Sample images for validation if dataset is large
        if len(ms_df) > self.config.get('sample_size', 1000):
            ms_df = ms_df.sample(
                min(self.config['sample_size'], len(ms_df)),
                random_state=42
            )
        
        # Validate each MS image
        ms_paths = [self.validator.data_dir / Path(p) for p in ms_df['ms_path']]
        for ms_path in tqdm(ms_paths, desc="Validating MS images"):
            if not ms_path.exists():
                results['ms_checks']['invalid'].append({
                    'path': str(ms_path),
                    'error': 'File not found'
                })
                continue
                
            validation = self._validate_ms_image(ms_path)
            if validation['valid']:
                results['ms_checks']['valid'].append({
                    'path': str(ms_path),
                    'band_count': validation['band_count']
                })
                # Update band distribution
                bc = validation['band_count']
                results['band_distribution'][bc] = results['band_distribution'].get(bc, 0) + 1
            else:
                results['ms_checks']['invalid'].append({
                    'path': str(ms_path),
                    'error': validation.get('error', 'Unknown error')
                })
        
        # Generate summary statistics
        total_checked = len(ms_df)
        valid_count = len(results['ms_checks']['valid'])
        invalid_count = len(results['ms_checks']['invalid'])
        
        results['summary'] = {
            'total_checked': total_checked,
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'valid_ratio': valid_count / total_checked if total_checked > 0 else 0,
            'band_distribution': results['band_distribution']
        }
        
        # Log warnings for common issues
        if invalid_count > 0:
            msg = f"Found {invalid_count} invalid MS images ({(invalid_count/total_checked)*100:.1f}%)"
            logger.warning(msg)
            results['warnings'].append(msg)
            
        if not results['band_distribution']:
            msg = "No valid MS images found for band distribution analysis"
            logger.warning(msg)
            results['warnings'].append(msg)
        
        return results


if __name__ == "__main__":
    main()