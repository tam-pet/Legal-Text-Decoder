"""
Data preprocessing script for Legal Text Decoder project.
Downloads, processes, and prepares the annotated legal text data.
"""

import os
import sys
import json
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import re

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_URL, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    TRAIN_FOLDER, TEST_FOLDER, RANDOM_SEED, LOG_DIR
)
from utils import (
    setup_logger, log_config, clean_text, extract_text_features,
    load_json_annotations, parse_label_studio_export,
    calculate_consensus_label, set_seed
)


def download_data(url: str, output_dir: Path, logger) -> Path:
    """
    Download data from SharePoint URL.
    
    Args:
        url: Download URL
        output_dir: Directory to save downloaded file
        logger: Logger instance
        
    Returns:
        Path to downloaded file
    """
    logger.info(f"Downloading data from: {url}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "legaltextdecoder.zip"
    
    # Check if already downloaded
    if output_path.exists():
        logger.info(f"Data already exists at {output_path}")
        return output_path
    
    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Please download manually from the SharePoint link and place in data/raw/")
        raise


def extract_data(zip_path: Path, output_dir: Path, logger) -> Path:
    """
    Extract downloaded zip file.
    
    Args:
        zip_path: Path to zip file
        output_dir: Directory to extract to
        logger: Logger instance
        
    Returns:
        Path to extracted directory
    """
    logger.info(f"Extracting {zip_path}")
    
    extracted_dir = output_dir / "legaltextdecoder"
    
    if extracted_dir.exists():
        logger.info(f"Data already extracted at {extracted_dir}")
        return extracted_dir
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    logger.info(f"Extracted to: {output_dir}")
    
    # Find the legaltextdecoder folder
    for item in output_dir.iterdir():
        if item.is_dir() and 'legaltextdecoder' in item.name.lower():
            return item
    
    return output_dir


def find_json_files(directory: Path) -> List[Path]:
    """Find all JSON files in a directory."""
    json_files = []
    for file_path in directory.rglob("*.json"):
        json_files.append(file_path)
    return json_files


def load_training_data(data_dir: Path, neptun_code: str, logger) -> List[Tuple[str, int]]:
    """
    Load training data from specified Neptun code folder.
    
    Args:
        data_dir: Base data directory
        neptun_code: Neptun code folder name
        logger: Logger instance
        
    Returns:
        List of (text, rating) tuples
    """
    logger.info(f"Loading training data from {neptun_code} folder")
    
    train_dir = data_dir / neptun_code
    
    if not train_dir.exists():
        # Try to find it in subdirectories
        for subdir in data_dir.rglob(neptun_code):
            if subdir.is_dir():
                train_dir = subdir
                break
    
    if not train_dir.exists():
        logger.warning(f"Training folder {neptun_code} not found")
        return []
    
    json_files = find_json_files(train_dir)
    logger.info(f"Found {len(json_files)} JSON files in training folder")
    
    all_data = []
    for json_file in json_files:
        try:
            data = load_json_annotations(json_file)
            parsed = parse_label_studio_export(data)
            all_data.extend(parsed)
            logger.info(f"  Loaded {len(parsed)} samples from {json_file.name}")
        except Exception as e:
            logger.warning(f"  Failed to load {json_file.name}: {e}")
    
    logger.info(f"Total training samples: {len(all_data)}")
    return all_data

def load_test_data(data_dir: Path, test_folder: str, logger) -> List[Tuple[str, int]]:
    """
    Load test data from all annotators - each annotation is a separate sample.
    No grouping or consensus calculation - every single annotation is evaluated independently.
    
    Args:
        data_dir: Base data directory
        test_folder: Test folder name (consensus folder with all annotator files)
        logger: Logger instance
        
    Returns:
        List of (text, rating) tuples - one per annotation
    """
    logger.info(f"Loading test data from {test_folder} folder")
    
    test_dir = data_dir / test_folder
    
    if not test_dir.exists():
        # Try to find it in subdirectories
        for subdir in data_dir.rglob(test_folder):
            if subdir.is_dir():
                test_dir = subdir
                break
    
    if not test_dir.exists():
        logger.warning(f"Test folder {test_folder} not found")
        return []
    
    json_files = find_json_files(test_dir)
    logger.info(f"Found {len(json_files)} JSON files in test folder")
    
    # Collect ALL annotations - no grouping, each annotation is a separate sample
    all_data = []
    
    for json_file in json_files:
        try:
            data = load_json_annotations(json_file)
            parsed = parse_label_studio_export(data)
            
            if not parsed:
                logger.warning(f"  {json_file.name} contains no valid annotations, skipping")
                continue
            
            all_data.extend(parsed)
            logger.info(f"  Loaded {len(parsed)} samples from {json_file.name}")
        except Exception as e:
            logger.warning(f"  Failed to load {json_file.name}: {e}")
    
    logger.info(f"\nTotal test samples: {len(all_data)}")
    
    return all_data

def analyze_data(train_data: List[Tuple[str, int]], 
                 test_data: List[Tuple[str, int]], 
                 logger) -> Dict[str, Any]:
    """
    Analyze the dataset and log statistics.
    
    Args:
        train_data: Training data (text, label) tuples
        test_data: Test data (text, label) tuples
        logger: Logger instance
        
    Returns:
        Dictionary of statistics
    """
    logger.info("=" * 60)
    logger.info("DATA ANALYSIS")
    logger.info("=" * 60)
    
    stats = {}
    
    # Training data analysis
    if train_data:
        train_texts, train_labels = zip(*train_data)
        
        train_label_dist = Counter(train_labels)
        stats['train_size'] = len(train_data)
        stats['train_label_distribution'] = dict(sorted(train_label_dist.items()))
        
        logger.info("\n--- Training Data ---")
        logger.info(f"Number of samples: {len(train_data)}")
        logger.info("Label distribution:")
        for label in sorted(train_label_dist.keys()):
            count = train_label_dist[label]
            pct = count / len(train_data) * 100
            logger.info(f"  Rating {label}: {count} ({pct:.1f}%)")
        
        # Text statistics
        text_lengths = [len(t.split()) for t in train_texts]
        stats['train_avg_words'] = np.mean(text_lengths)
        stats['train_min_words'] = np.min(text_lengths)
        stats['train_max_words'] = np.max(text_lengths)
        
        logger.info(f"Text length (words): mean={stats['train_avg_words']:.1f}, "
                    f"min={stats['train_min_words']}, max={stats['train_max_words']}")
    
    # Test data analysis
    if test_data:
        test_texts, test_labels = zip(*test_data)
        
        test_label_dist = Counter(test_labels)
        stats['test_size'] = len(test_data)
        stats['test_label_distribution'] = dict(sorted(test_label_dist.items()))
        
        logger.info("\n--- Test Data (All Annotations) ---")
        logger.info(f"Number of samples: {len(test_data)}")
        logger.info("Label distribution:")
        for label in sorted(test_label_dist.keys()):
            count = test_label_dist[label]
            pct = count / len(test_data) * 100
            logger.info(f"  Rating {label}: {count} ({pct:.1f}%)")
        
        # Text statistics
        text_lengths = [len(t.split()) for t in test_texts]
        stats['test_avg_words'] = np.mean(text_lengths)
        stats['test_min_words'] = np.min(text_lengths)
        stats['test_max_words'] = np.max(text_lengths)
        logger.info(f"Text length (words): mean={stats['test_avg_words']:.1f}, "
                    f"min={stats['test_min_words']}, max={stats['test_max_words']}")
    
    logger.info("=" * 60)
    
    return stats


def save_processed_data(train_data: List[Tuple[str, int]],
                        test_data: List[Tuple[str, int]],
                        output_dir: Path,
                        logger) -> None:
    """
    Save processed data to CSV files.
    
    Args:
        train_data: Training data (text, label) tuples
        test_data: Test data (text, label) tuples
        output_dir: Output directory
        logger: Logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training data
    if train_data:
        train_df = pd.DataFrame(train_data, columns=['text', 'label'])
        train_df['label'] = train_df['label'].astype(int)
        train_path = output_dir / "train.csv"
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        logger.info(f"Saved training data to {train_path}")
    
    # Save test data
    if test_data:
        test_df = pd.DataFrame(test_data, columns=['text', 'label'])
        test_df['label'] = test_df['label'].astype(int)
        test_path = output_dir / "test.csv"
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        logger.info(f"Saved test data to {test_path}")
    
    # Save combined statistics
    stats = {
        'train_samples': len(train_data) if train_data else 0,
        'test_samples': len(test_data) if test_data else 0,
    }
    stats_path = output_dir / "data_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")


def main():
    """Main preprocessing pipeline."""
    # Setup logger
    logger = setup_logger(
        name="DataPreprocessing",
        log_file=LOG_DIR / "run.log",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("LEGAL TEXT DECODER - DATA PREPROCESSING")
    logger.info("=" * 60)
    
    set_seed(RANDOM_SEED)
    
    # Log configuration
    log_config(logger, {
        "data_url": DATA_URL,
        "train_folder": TRAIN_FOLDER,
        "test_folder": TEST_FOLDER,
        "raw_data_dir": str(RAW_DATA_DIR),
        "processed_data_dir": str(PROCESSED_DATA_DIR),
    }, title="Data Configuration")
    
    try:
        # Step 1: Download data
        logger.info("\n[Step 1] Downloading data...")
        zip_path = download_data(DATA_URL, RAW_DATA_DIR, logger)
        
        # Step 2: Extract data
        logger.info("\n[Step 2] Extracting data...")
        data_dir = extract_data(zip_path, RAW_DATA_DIR, logger)
        
        # Step 3: Load training data
        logger.info("\n[Step 3] Loading training data...")
        train_data = load_training_data(data_dir, TRAIN_FOLDER, logger)
        
        # Step 4: Load test data (all annotations)
        logger.info("\n[Step 4] Loading test data (all annotations)...")
        test_data = load_test_data(data_dir, TEST_FOLDER, logger)
        
        # Step 5: Analyze data
        logger.info("\n[Step 5] Analyzing data...")
        stats = analyze_data(train_data, test_data, logger)
        
        # Step 6: Save processed data
        logger.info("\n[Step 6] Saving processed data...")
        save_processed_data(train_data, test_data, PROCESSED_DATA_DIR, logger)
        
        logger.info("\n" + "=" * 60)
        logger.info("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
