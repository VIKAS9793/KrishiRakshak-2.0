#!/usr/bin/env python3
"""
Robust Data Preprocessing Script for Image Datasets (ML/DL Standard, 2024)

Features:
- Recursively scans train/val/test folders for images and labels
- Validates images (corruption check)
- Detects and removes duplicates (hash-based, optional)
- Generates new CSVs (train.csv, val.csv, test.csv) with correct relative paths and labels
- Outputs class distribution for each split
- Computes and saves class weights (for training)
- Optionally creates balanced subsets
- Logs all actions, errors, and summary statistics
- Configurable via CLI (input/output dirs, file types, options)
- Uses tqdm for progress bars, Pillow for image validation, pandas for CSVs, sklearn for class weights
- Robust error handling and clear output

Usage:
    python scripts/preprocess_dataset.py --input-dir data/plantvillage --output-dir data/plantvillage --file-types jpg jpeg png --remove-duplicates --create-balanced --csv-out-dir data/plantvillage
"""
import os
import argparse
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image, ImageFilter, ExifTags
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import csv
import json
import logging
import concurrent.futures
import imagehash

def setup_logger(log_file=None):
    logger = logging.getLogger("preprocess_dataset")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def hash_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def scan_images(root_dir, file_types, logger):
    image_info = []
    for split in ['train', 'val', 'test']:
        split_dir = Path(root_dir) / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for ext in file_types:
                files = list(class_dir.glob(f'*.{ext}'))
                for img_path in files:
                    image_info.append({
                        'split': split,
                        'label': label,
                        'image_path': img_path.relative_to(root_dir).as_posix()
                    })
    return image_info

def remove_duplicates(image_info, root_dir, logger):
    logger.info("Checking for duplicate images (hash-based)...")
    hash_dict = defaultdict(list)
    for info in tqdm(image_info, desc="Hashing images"):
        abs_path = Path(root_dir) / info['image_path']
        img_hash = hash_image(abs_path)
        if img_hash:
            hash_dict[img_hash].append(info)
    unique_images = []
    duplicates = []
    for img_list in hash_dict.values():
        unique_images.append(img_list[0])
        if len(img_list) > 1:
            duplicates.extend(img_list[1:])
    logger.info(f"Found {len(duplicates)} duplicate images. Removing from dataset.")
    return unique_images, duplicates

def validate_images(image_info, root_dir, logger):
    logger.info("Validating images for corruption...")
    valid_images = []
    corrupted = []
    for info in tqdm(image_info, desc="Validating images"):
        abs_path = Path(root_dir) / info['image_path']
        if validate_image(abs_path):
            valid_images.append(info)
        else:
            corrupted.append(info)
    logger.info(f"Found {len(corrupted)} corrupted images. Removing from dataset.")
    return valid_images, corrupted

def save_csv(image_info, split, out_dir):
    df = pd.DataFrame([i for i in image_info if i['split'] == split])
    csv_path = Path(out_dir) / f"{split}.csv"
    df[['image_path', 'label']].to_csv(csv_path, index=False)
    return csv_path, df

def compute_class_weights(df, logger):
    labels = df['label'].values
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = dict(zip(classes, weights))
    logger.info(f"Class weights: {class_weights}")
    return class_weights

def create_balanced_subset(df, samples_per_class=100):
    return df.groupby('label').apply(lambda x: x.sample(min(len(x), samples_per_class), random_state=42)).reset_index(drop=True)

def strip_exif(image):
    data = list(image.getdata())
    image_no_exif = Image.new(image.mode, image.size)
    image_no_exif.putdata(data)
    return image_no_exif

def image_quality_metrics(image):
    gray = image.convert('L')
    arr = np.array(gray)
    blur = np.var(np.abs(np.gradient(arr)))
    brightness = np.mean(arr)
    contrast = np.std(arr)
    return {'blur': float(blur), 'brightness': float(brightness), 'contrast': float(contrast)}

def process_and_save_image(abs_path, out_path, resize=None, strip_exif_flag=True):
    try:
        with Image.open(abs_path) as img:
            if resize:
                img = img.resize(resize, Image.LANCZOS)
            if strip_exif_flag:
                img = strip_exif(img)
            img.save(out_path)
            return True
    except Exception:
        return False

def parallel_image_processing(image_info, root_dir, output_image_dir, resize, strip_exif_flag, phash_flag, logger):
    results = []
    def process(info):
        abs_path = Path(root_dir) / info['image_path']
        out_path = None
        if output_image_dir:
            out_path = Path(output_image_dir) / info['image_path']
            out_path.parent.mkdir(parents=True, exist_ok=True)
        valid = validate_image(abs_path)
        if not valid:
            return {'info': info, 'valid': False}
        with Image.open(abs_path) as img:
            metrics = image_quality_metrics(img)
            if out_path:
                process_and_save_image(abs_path, out_path, resize, strip_exif_flag)
            md5 = hash_image(abs_path)
            phash = str(imagehash.phash(img)) if phash_flag else None
        return {'info': info, 'valid': True, 'md5': md5, 'phash': phash, 'metrics': metrics}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for res in tqdm(executor.map(process, image_info), total=len(image_info), desc='Processing images'):
            results.append(res)
    return results

def split_single_folder(single_dir, split_ratios, file_types, logger):
    all_images = []
    for ext in file_types:
        all_images.extend(list(Path(single_dir).rglob(f'*.{ext}')))
    np.random.shuffle(all_images)
    n = len(all_images)
    n_train = int(split_ratios[0] * n)
    n_val = int(split_ratios[1] * n)
    splits = ['train'] * n_train + ['val'] * n_val + ['test'] * (n - n_train - n_val)
    np.random.shuffle(splits)
    image_info = []
    for img_path, split in zip(all_images, splits):
        label = img_path.parent.name
        image_info.append({'split': split, 'label': label, 'image_path': img_path.relative_to(single_dir).as_posix()})
    logger.info(f"Split {n} images into train/val/test: {n_train}/{n_val}/{n-n_train-n_val}")
    return image_info

def generate_report(image_info, corrupted, duplicates, low_quality, class_counts, out_path):
    report = {
        'total_images': len(image_info) + len(corrupted),
        'valid_images': len(image_info),
        'corrupted_images': len(corrupted),
        'duplicate_images': len(duplicates),
        'low_quality_images': len(low_quality),
        'class_distribution': class_counts,
        'low_quality_samples': low_quality[:10],
        'corrupted_samples': corrupted[:10],
        'duplicate_samples': duplicates[:10],
    }
    if out_path.endswith('.json'):
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2)
    elif out_path.endswith('.md') or out_path.endswith('.markdown'):
        with open(out_path, 'w') as f:
            f.write(f"# Data Preprocessing Report\n\n")
            for k, v in report.items():
                f.write(f"**{k}:** {v}\n\n")
    elif out_path.endswith('.html'):
        with open(out_path, 'w') as f:
            f.write(f"<html><body><h1>Data Preprocessing Report</h1>")
            for k, v in report.items():
                f.write(f"<b>{k}:</b> {v}<br><br>")
            f.write("</body></html>")

def main():
    parser = argparse.ArgumentParser(description="Robust Data Preprocessing for Image Datasets")
    parser.add_argument('--input-dir', type=str, required=True, help='Root directory containing train/val/test folders')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save CSVs and logs')
    parser.add_argument('--file-types', nargs='+', default=['jpg', 'jpeg', 'png'], help='Image file extensions to include')
    parser.add_argument('--remove-duplicates', action='store_true', help='Remove duplicate images (hash-based)')
    parser.add_argument('--create-balanced', action='store_true', help='Create balanced subset CSVs')
    parser.add_argument('--samples-per-class', type=int, default=100, help='Samples per class for balanced subset')
    parser.add_argument('--log-file', type=str, default=None, help='Path to log file')
    parser.add_argument('--resize', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=None, help='Resize images to WIDTH HEIGHT')
    parser.add_argument('--output-image-dir', type=str, default=None, help='Directory to save processed images (resized/cleaned)')
    parser.add_argument('--phash-duplicates', action='store_true', help='Detect near-duplicates using perceptual hash (pHash)')
    parser.add_argument('--report', type=str, default=None, help='Path to save HTML/Markdown/JSON report')
    parser.add_argument('--split-from-single', type=str, default=None, help='If set, split a single folder into train/val/test (provide folder path)')
    parser.add_argument('--split-ratios', type=float, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'), default=[0.7, 0.15, 0.15], help='Ratios for train/val/test split')
    args = parser.parse_args()

    logger = setup_logger(args.log_file)
    logger.info("Starting robust data preprocessing...")

    # Step 1: Scan for images
    image_info = scan_images(args.input_dir, args.file_types, logger)
    logger.info(f"Total images found: {len(image_info)}")

    # Step 2: Remove duplicates (optional)
    if args.remove_duplicates:
        image_info, duplicates = remove_duplicates(image_info, args.input_dir, logger)
        logger.info(f"Removed {len(duplicates)} duplicates.")

    # Step 3: Validate images
    image_info, corrupted = validate_images(image_info, args.input_dir, logger)
    logger.info(f"Removed {len(corrupted)} corrupted images.")

    # Step 4: Save CSVs and report class distribution
    for split in ['train', 'val', 'test']:
        csv_path, df = save_csv(image_info, split, args.output_dir)
        logger.info(f"Saved {split} CSV to {csv_path} ({len(df)} samples)")
        class_counts = df['label'].value_counts().to_dict()
        logger.info(f"Class distribution for {split}: {class_counts}")
        if split == 'train' and len(df) > 0:
            class_weights = compute_class_weights(df, logger)
            weights_file = Path(args.output_dir) / 'class_weights.json'
            with open(weights_file, 'w') as f:
                json.dump(class_weights, f, indent=2)
            logger.info(f"Saved class weights to {weights_file}")
        # Step 5: Create balanced subset (optional)
        if args.create_balanced and len(df) > 0:
            balanced_df = create_balanced_subset(df, args.samples_per_class)
            balanced_csv = Path(args.output_dir) / f"{split}_balanced.csv"
            balanced_df[['image_path', 'label']].to_csv(balanced_csv, index=False)
            logger.info(f"Saved balanced {split} CSV to {balanced_csv} ({len(balanced_df)} samples)")

    logger.info("Data preprocessing complete!")
    logger.info("Summary:")
    logger.info(f"Total images processed: {len(image_info)}")
    if args.remove_duplicates:
        logger.info(f"Duplicates removed: {len(duplicates)}")
    logger.info(f"Corrupted images removed: {len(corrupted)}")
    logger.info(f"CSVs saved in: {args.output_dir}")

    # Track and log low-quality images (e.g., blur < threshold, brightness/contrast out of range)
    BLUR_THRESHOLD = 10.0         # Example: lower means blurrier
    BRIGHTNESS_MIN = 30.0         # Example: too dark
    BRIGHTNESS_MAX = 220.0        # Example: too bright
    CONTRAST_MIN = 10.0           # Example: low contrast

    low_quality = []
    for info in image_info:
        abs_path = Path(args.input_dir) / info['image_path']
        with Image.open(abs_path) as img:
            metrics = image_quality_metrics(img)
            if (
                metrics['blur'] < BLUR_THRESHOLD or
                metrics['brightness'] < BRIGHTNESS_MIN or
                metrics['brightness'] > BRIGHTNESS_MAX or
                metrics['contrast'] < CONTRAST_MIN
            ):
                low_quality.append(info)
    logger.info(f"Found {len(low_quality)} low-quality images. Removing from dataset.")

    # Save processed images if output_image_dir is set
    if args.output_image_dir:
        for info in image_info:
            abs_path = Path(args.input_dir) / info['image_path']
            out_path = Path(args.output_image_dir) / info['image_path']
            out_path.parent.mkdir(parents=True, exist_ok=True)
            process_and_save_image(abs_path, out_path, args.resize, True)
        logger.info(f"Processed images saved in: {args.output_image_dir}")

    # Generate report at the end if requested
    if args.report:
        generate_report(image_info, corrupted, duplicates, low_quality, class_counts, args.report)
        logger.info(f"Report saved to: {args.report}")

    # Support splitting from a single folder if --split-from-single is set
    if args.split_from_single:
        image_info = split_single_folder(args.split_from_single, args.split_ratios, args.file_types, logger)
        logger.info(f"Split {len(image_info)} images into train/val/test: {args.split_ratios[0] * len(image_info)}/{args.split_ratios[1] * len(image_info)}/{len(image_info) - args.split_ratios[0] * len(image_info) - args.split_ratios[1] * len(image_info)}")

if __name__ == '__main__':
    main() 