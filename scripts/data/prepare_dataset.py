"""
Dataset Preparation Script for KrishiSahayak.

This script processes both PlantDoc and PlantVillage datasets, creating a unified
metadata.csv file with image paths, labels, and stratified train/val/test splits.
"""
import argparse
import logging
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

class DatasetProcessor:
    """Processes image datasets and creates train/val/test splits."""
    
    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.config = config
        self.split_ratios = config['data']['split_ratios']
        self.min_samples = config['data']['min_samples_per_class']
        self.processed_dir = Path(config['data']['processed_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def process_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Process a single dataset."""
        dataset_cfg = self.config['data']['datasets'][dataset_name]
        dataset_path = Path(dataset_cfg['path'])
        
        logger.info(f"Processing {dataset_name} dataset from {dataset_path}")
        
        # Collect all image files and their labels
        samples = []
        for class_dir in tqdm(list(dataset_path.rglob('*')), desc=f"Scanning {dataset_name}"):
            if not class_dir.is_dir():
                continue
                
            class_name = f"{dataset_name}_{class_dir.name}"  # Prefix with dataset name
            
            # Count images in this class
            image_paths = [
                p for p in class_dir.glob('*')
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            
            if len(image_paths) < self.min_samples:
                logger.debug(f"Skipping class {class_name}: only {len(image_paths)} samples (min {self.min_samples} required)")
                continue
                
            for img_path in image_paths:
                samples.append({
                    'image_path': str(img_path.relative_to(dataset_path.parent)),
                    'label': class_name,
                    'dataset': dataset_name,
                    'class_dir': class_dir.name
                })
        
        if not samples:
            logger.warning(f"No valid samples found for {dataset_name}")
            return pd.DataFrame()
            
        return pd.DataFrame(samples)
    
    def create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create train/val/test splits with stratification."""
        if df.empty:
            return {}
            
        # First split: train vs (val + test)
        train_ratio = self.split_ratios['train']
        val_test_ratio = self.split_ratios['val'] + self.split_ratios['test']
        
        train_df, temp_df = train_test_split(
            df,
            test_size=val_test_ratio,
            random_state=42,
            stratify=df['label']
        )
        
        # Second split: val vs test
        val_ratio = self.split_ratios['val'] / val_test_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_ratio,  # Because we're using the remaining portion
            random_state=42,
            stratify=temp_df['label']
        )
        
        # Add split information
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def save_metadata(self, df: pd.DataFrame, dataset_name: str):
        """Save dataset metadata to CSV."""
        if df.empty:
            return
            
        output_path = self.processed_dir / f"{dataset_name}_metadata.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} samples to {output_path}")
    
    def run(self):
        """Process all datasets, combine them, then create unified splits."""
        # Step 1: Collect all data from all datasets
        all_data = []
        
        for dataset_name in self.config['data']['datasets']:
            logger.info(f"Collecting data from {dataset_name}...")
            df = self.process_dataset(dataset_name)
            if not df.empty:
                all_data.append(df)
                logger.info(f"  - Collected {len(df)} samples from {dataset_name}")
        
        if not all_data:
            raise ValueError("No valid data found in any dataset")
        
        # Step 2: Combine all data into a single DataFrame
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"\nCombined dataset: {len(combined_df)} total samples from {combined_df['dataset'].nunique()} datasets")
        
        # Step 3: Create unified splits (merge-then-split strategy)
        logger.info("\nCreating unified train/val/test splits...")
        
        # First split: train vs (val + test)
        train_ratio = self.split_ratios['train']
        val_test_ratio = self.split_ratios['val'] + self.split_ratios['test']
        
        train_df, temp_df = train_test_split(
            combined_df,
            test_size=val_test_ratio,
            random_state=42,
            stratify=combined_df['label']
        )
        
        # Second split: val vs test
        val_ratio = self.split_ratios['val'] / val_test_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_ratio,
            random_state=42,
            stratify=temp_df['label']
        )
        
        # Add split information
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        # Combine back into final DataFrame
        final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # Save metadata
        self._save_metadata_and_stats(final_df)
        
        return final_df
    
    def _save_metadata_and_stats(self, df: pd.DataFrame):
        """Save metadata and print dataset statistics."""
        # Save combined metadata
        combined_path = self.processed_dir / 'metadata.csv'
        df.to_csv(combined_path, index=False)
        logger.info(f"\nSaved metadata to {combined_path}")
        
        # Print dataset statistics
        logger.info("\n=== Final Dataset Statistics ===")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Number of unique classes: {df['label'].nunique()}")
        
        # Samples per split
        split_counts = df['split'].value_counts()
        logger.info("\nSamples per split:")
        for split in ['train', 'val', 'test']:
            count = split_counts.get(split, 0)
            pct = (count / len(df)) * 100
            logger.info(f"  - {split}: {count} samples ({pct:.1f}%)")
        
        # Samples per dataset
        logger.info("\nSamples per dataset:")
        for dataset, count in df['dataset'].value_counts().items():
            pct = (count / len(df)) * 100
            logger.info(f"  - {dataset}: {count} samples ({pct:.1f}%)")
        
        # Class distribution per split
        logger.info("\nClass distribution across splits:")
        for split in ['train', 'val', 'test']:
            split_df = df[df['split'] == split]
            logger.info(f"\n  {split.upper()} SET - {len(split_df)} samples:")
            logger.info(f"  - Unique classes: {split_df['label'].nunique()}")
            logger.info(f"  - Samples/class: {split_df['label'].value_counts().describe().to_dict()}")

def create_metadata(raw_data_dir: Path, output_dir: Path, train_ratio: float, val_ratio: float):
    """
    Scans the raw data directory, creates stratified splits, and saves a metadata CSV.
    
    Args:
        raw_data_dir (Path): The path to the directory containing class-named subfolders of images.
        output_dir (Path): The directory where the 'metadata.csv' will be saved.
        train_ratio (float): The proportion of the dataset to allocate for training.
        val_ratio (float): The proportion of the dataset to allocate for validation.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    filepaths = []

    logger.info(f"Scanning for images in '{raw_data_dir}'...")
    # Using pathlib for modern, object-oriented path handling
    image_paths_generator = raw_data_dir.rglob('*')
    
    # First pass: count images per class
    class_counts = {}
    for path in tqdm(list(image_paths_generator), desc="Counting images"):
        if path.is_file() and path.suffix.lower() in image_extensions:
            label = path.parent.name
            class_counts[label] = class_counts.get(label, 0) + 1
    
    # Second pass: collect filepaths and ensure minimum samples per class
    min_samples = 5  # Minimum samples per class to be included
    valid_classes = {k for k, v in class_counts.items() if v >= min_samples}
    
    if not valid_classes:
        raise ValueError(f"No classes with at least {min_samples} samples found.")
    
    logger.info(f"Found {len(valid_classes)} classes with at least {min_samples} samples.")
    
    # Collect filepaths for valid classes only
    image_paths_generator = raw_data_dir.rglob('*')
    for path in tqdm(list(image_paths_generator), desc="Processing images"):
        if path.is_file() and path.suffix.lower() in image_extensions:
            label = path.parent.name
            if label in valid_classes:
                relative_path = path.relative_to(raw_data_dir)
                filepaths.append({'image_path': str(relative_path), 'label': label})

    if not filepaths:
        raise FileNotFoundError(f"No valid images found in {raw_data_dir}. "
                              f"Ensure there are classes with at least {min_samples} samples.")

    logger.info(f"Found {len(filepaths)} total images across {len(valid_classes)} classes.")
    
    # Create DataFrame and ensure we have enough samples for splitting
    df = pd.DataFrame(filepaths)
    
    # Adjust split ratios if necessary to ensure at least 1 sample per class in each split
    class_sizes = df['label'].value_counts()
    min_class_size = class_sizes.min()
    
    # Calculate minimum ratio needed to have at least 1 sample in validation
    min_val_ratio = 1.0 / min_class_size
    if val_ratio < min_val_ratio:
        logger.warning(f"Adjusted val_ratio from {val_ratio} to {min_val_ratio} "
                      f"to ensure at least 1 sample per class in validation")
        val_ratio = min_val_ratio
        
    # Ensure train ratio is valid
    train_ratio = max(0.1, min(0.9, train_ratio))  # Keep between 0.1 and 0.9
    
    # Adjust ratios to sum to 1
    total_ratio = train_ratio + val_ratio
    train_ratio = train_ratio / total_ratio
    val_ratio = val_ratio / total_ratio
    
    logger.info(f"Using train/val split: {train_ratio:.2f}/{val_ratio:.2f}")
    
    # Split into train and validation
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=42,
        stratify=df['label']
    )
    
    # Add split information
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    # For now, we'll use the same data for test as validation
    # since we have a separate test directory
    test_df = val_df.copy()
    test_df['split'] = 'test'  # This will be overridden by the separate test directory

    # Combine the splits into a single dataframe
    final_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    # Save the final metadata file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'metadata.csv'
    final_df.to_csv(output_path, index=False)
    
    logger.info(f"Successfully created metadata file at '{output_path}'")
    logger.info("Split distribution:\n" + str(final_df['split'].value_counts()))


def main():
    """Main entry point to parse args, load config, and start preparation."""
    parser = argparse.ArgumentParser(description='Prepare dataset metadata from multiple datasets.')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to the project configuration YAML file.')
    parser.add_argument('--output-dir', type=str, 
                       help='Directory to save metadata. Overrides config if provided.')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {args.config}")
        return
    
    # Override output directory if provided
    if args.output_dir:
        config['data']['processed_dir'] = args.output_dir
    
    # Create and run the dataset processor
    try:
        processor = DatasetProcessor(config)
        processor.run()
        logger.info("Dataset preparation completed successfully!")
    except Exception as e:
        logger.error(f"Error during dataset preparation: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
