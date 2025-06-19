#!/usr/bin/env python3
"""Fix dataset issues for KrishiSahayak.

1. Updates file paths in CSV to match current directory structure
2. Handles class imbalance by setting up class weights
3. Creates a balanced subset if needed

Usage:
    python scripts/fix_dataset.py --input-dir data/processed_data --output-dir data/fixed_data
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def fix_file_paths(csv_path: Path, base_dir: Path) -> pd.DataFrame:
    """Update file paths in CSV to be relative to the project root."""
    df = pd.read_csv(csv_path)
    
    # Convert to absolute paths relative to project root
    df['image_path'] = df['image_path'].apply(
        lambda x: str(base_dir / '/'.join(Path(x).parts[-3:]))
    )
    return df

def get_class_weights(df: pd.DataFrame) -> dict:
    """Calculate class weights to handle imbalance."""
    # Get class counts
    class_counts = df['label'].value_counts()
    total_samples = len(df)
    num_classes = len(class_counts)
    
    # Calculate weights inversely proportional to class frequencies
    class_weights = {}
    for label, count in class_counts.items():
        class_weights[label] = total_samples / (num_classes * count)
        
    return class_weights

def create_balanced_subset(df: pd.DataFrame, samples_per_class: int = 100) -> pd.DataFrame:
    """Create a balanced subset of the data."""
    return df.groupby('label').apply(
        lambda x: x.sample(min(len(x), samples_per_class), random_state=42)
    ).reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description="Fix dataset issues for KrishiSahayak")
    parser.add_argument('--input-dir', type=Path, default='data/processed_data',
                      help='Directory containing train/val/test CSVs')
    parser.add_argument('--output-dir', type=Path, default='data/fixed_data',
                      help='Directory to save fixed CSVs')
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base directory where images are actually stored
    base_dir = Path('data/plantvillage')
    
    # Process each split
    for split in ['train', 'val', 'test']:
        input_csv = args.input_dir / f'{split}.csv'
        if not input_csv.exists():
            print(f"Warning: {input_csv} not found, skipping")
            continue
            
        # Fix file paths
        print(f"Processing {split} split...")
        df = fix_file_paths(input_csv, base_dir)
        
        # Save fixed CSV
        output_csv = args.output_dir / f'{split}.csv'
        df.to_csv(output_csv, index=False)
        print(f"Saved fixed {split} CSV to {output_csv}")
        
        # Calculate and display class distribution
        class_counts = Counter(df['label'])
        print(f"\nClass distribution for {split}:")
        for cls, count in class_counts.most_common():
            print(f"  {cls}: {count}")
        
        # Calculate class weights (useful for training)
        if split == 'train':
            class_weights = get_class_weights(df)
            print("\nClass weights for training:")
            for cls, weight in class_weights.items():
                print(f"  {cls}: {weight:.2f}")
            
            # Save class weights
            weights_file = args.output_dir / 'class_weights.json'
            import json
            with open(weights_file, 'w') as f:
                json.dump(class_weights, f, indent=2)
            print(f"\nSaved class weights to {weights_file}")
    
    print("\nDataset fixing complete!")
    print(f"Next steps:")
    print(f"1. Verify the first few rows of the fixed CSVs in {args.output_dir}")
    print(f"2. Use the class_weights.json during model training")
    print(f"3. Consider data augmentation for minority classes")

if __name__ == '__main__':
    main()
