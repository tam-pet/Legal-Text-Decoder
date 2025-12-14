"""
Utility functions and logging configuration for Legal Text Decoder project.
"""

import logging
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def setup_logger(
    name: str = "LegalTextDecoder",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up and configure logger for the project.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (stdout for Docker)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_config(logger: logging.Logger, config: Dict[str, Any], title: str = "Configuration") -> None:
    """Log configuration parameters."""
    logger.info("=" * 60)
    logger.info(f"{title}")
    logger.info("=" * 60)
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for k, v in value.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 60)


def log_model_summary(logger: logging.Logger, model, model_name: str = "Model") -> None:
    """Log model architecture summary."""
    logger.info("=" * 60)
    logger.info(f"{model_name} Architecture Summary")
    logger.info("=" * 60)
    
    # For PyTorch models
    if hasattr(model, 'parameters'):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {non_trainable_params:,}")
    
    # For sklearn models
    elif hasattr(model, 'get_params'):
        params = model.get_params()
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_metrics: Dict[str, float],
    val_loss: Optional[float] = None,
    val_metrics: Optional[Dict[str, float]] = None
) -> None:
    """Log training progress for an epoch."""
    msg = f"Epoch [{epoch}/{total_epochs}] | Train Loss: {train_loss:.4f}"
    
    for name, value in train_metrics.items():
        msg += f" | Train {name}: {value:.4f}"
    
    if val_loss is not None:
        msg += f" | Val Loss: {val_loss:.4f}"
    
    if val_metrics:
        for name, value in val_metrics.items():
            msg += f" | Val {name}: {value:.4f}"
    
    logger.info(msg)


def log_evaluation_results(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    title: str = "Evaluation Results"
) -> None:
    """Log evaluation metrics."""
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{name}: {value:.4f}")
        elif isinstance(value, np.ndarray):
            logger.info(f"{name}:")
            logger.info(f"\n{value}")
        else:
            logger.info(f"{name}: {value}")
    
    logger.info("=" * 60)


# =============================================================================
# Text Processing Utilities
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove special unicode characters but keep Hungarian characters
    text = re.sub(r'[^\w\s\-.,;:!?()\"\'áéíóöőúüűÁÉÍÓÖŐÚÜŰ]', '', text)
    
    return text


def extract_text_features(text: str) -> Dict[str, float]:
    """
    Extract basic text features for analysis.
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary of text features
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    features = {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "avg_sentence_length": np.mean([len(s.split()) for s in sentences]) if sentences else 0,
        "unique_word_ratio": len(set(words)) / len(words) if words else 0,
    }
    
    return features



# Jogi szakszavak listája (magyar)
LEGAL_KEYWORDS = [
    'szolgáltató', 'felhasználó', 'szerződés', 'ászf', 'adatkezelés',
    'jogosult', 'kötelezett', 'felmondás', 'megszüntetés', 'elállás',
    'kártérítés', 'garancia', 'szavatosság', 'jogorvoslat', 'polgári',
    'törvény', 'rendelet', 'hatályos', 'módosítás', 'érvényes',
    'jogosultság', 'felelősség', 'kizárólagos', 'fenntartja', 'értelmében',
    'rendelkezés', 'kikötés', 'feltétel', 'megállapodás', 'biztosít',
    'kötelezettség', 'igénybevétel', 'hatáskör', 'jogviszony', 'teljesítés'
]

def extract_advanced_features(text: str) -> Dict[str, float]:
    """
    Extract advanced linguistic features from text.
    
    Features:
    - word_count: Number of words
    - char_count: Number of characters
    - avg_word_length: Average word length
    - sentence_count: Number of sentences
    - legal_word_count: Count of legal keywords
    - legal_word_ratio: Ratio of legal keywords to total words
    - complex_word_ratio: Ratio of long words (>10 chars)
    - punctuation_ratio: Ratio of punctuation marks
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Basic counts
    words = text.split()
    features['word_count'] = len(words)
    features['char_count'] = len(text)
    
    # Average word length
    if words:
        features['avg_word_length'] = np.mean([len(w) for w in words])
    else:
        features['avg_word_length'] = 0
    
    # Sentence count (approximate)
    sentences = re.split(r'[.!?]+', text)
    features['sentence_count'] = len([s for s in sentences if s.strip()])
    
    # Legal keywords
    text_lower = text.lower()
    legal_count = sum(1 for keyword in LEGAL_KEYWORDS if keyword in text_lower)
    features['legal_word_count'] = legal_count
    features['legal_word_ratio'] = legal_count / len(words) if words else 0
    
    # Complex words (longer than 10 characters)
    complex_words = [w for w in words if len(w) > 10]
    features['complex_word_ratio'] = len(complex_words) / len(words) if words else 0
    
    # Punctuation ratio
    punctuation = re.findall(r'[,;:.!?()"\'-]', text)
    features['punctuation_ratio'] = len(punctuation) / len(text) if text else 0
    
    return features


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_json_annotations(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load annotations from a Label Studio JSON export file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of annotation dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def parse_label_studio_export(data: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """
    Parse Label Studio export format to extract text and ratings.
    
    Args:
        data: List of Label Studio annotation objects
        
    Returns:
        List of (text, rating) tuples
    """
    results = []
    
    for item in data:
        text = item.get('data', {}).get('text', '')
        annotations = item.get('annotations', [])
        
        if annotations and annotations[0].get('result'):
            result = annotations[0]['result']
            
            # Handle different annotation formats
            for r in result:
                value = r.get('value', {})
                
                # Rating format
                if 'rating' in value:
                    rating = value['rating']
                    results.append((text, int(rating)))
                    break
                
                # Choices format (from the XML config)
                elif 'choices' in value:
                    choices = value['choices']
                    if choices:
                        choice = choices[0]
                        # Extract rating number from choice text like "1-Nagyon nehezen érthető"
                        rating_match = re.match(r'^(\d)', choice)
                        if rating_match:
                            rating = int(rating_match.group(1))
                            results.append((text, rating))
                            break
    
    return results


def calculate_consensus_label(
    annotations: List[int],
    method: str = "majority"
) -> Tuple[int, float]:
    """
    Calculate consensus label from multiple annotations.
    
    Args:
        annotations: List of ratings from different annotators
        method: 'majority' for most common, 'mean' for average (rounded)
        
    Returns:
        Tuple of (consensus_label, agreement_score)
    """
    if not annotations:
        return None, 0.0
    
    if method == "majority":
        from collections import Counter
        counter = Counter(annotations)
        most_common = counter.most_common(1)[0]
        consensus = most_common[0]
        agreement = most_common[1] / len(annotations)
    elif method == "mean":
        consensus = int(round(np.mean(annotations)))
        # Calculate agreement as inverse of std deviation
        std = np.std(annotations)
        agreement = max(0, 1 - std / 2)  # Normalize to 0-1
    else:
        raise ValueError(f"Unknown consensus method: {method}")
    
    return consensus, agreement


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# =============================================================================
# Metrics Utilities
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report, mean_absolute_error,
        cohen_kappa_score
    )
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "mae": mean_absolute_error(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }
    
    return metrics
