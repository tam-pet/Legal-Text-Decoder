"""
Configuration file for Legal Text Decoder project.
Contains all hyperparameters, paths, and settings.
"""

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "log"

# Data URLs
DATA_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1"

# Training data folder (your Neptun code)
TRAIN_FOLDER = "FA0B9B"
# Test data folder (consensus annotations)
TEST_FOLDER = "consensus"





# =============================================================================
# Model Hyperparameters
# =============================================================================

# General
RANDOM_SEED = 42
NUM_CLASSES = 5  # Ratings 1-5

# Baseline Model (TF-IDF + Classifier)
BASELINE_CONFIG = {
    'classifier': 'logistic_regression',  # Vagy 'random_forest', 'gradient_boosting'
    'class_weight': 'balanced',
    
    'tfidf_max_features': 1500,  
    'tfidf_ngram_range': (1, 3),  # Csak unigram + bigram, trigram túl sok
    'tfidf_min_df': 1,  # ← CSÖKKENTSD 2-ről 1-re (ne veszíts ritka szavakat!)
    'tfidf_max_df': 0.9,  # ← CSÖKKENTSD 0.95-ről (agresszívebb stop word szűrés)
    
    'use_augmentation': False,
}

# Simple MLP Model (Neural Network on TF-IDF features)
MLP_CONFIG = {
    "tfidf_max_features": 3000,
    "tfidf_ngram_range": (1, 3),
    "hidden_layers": [512, 256, 128],  # 3-layer MLP
    "dropout": 0.4,
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 50,
    "early_stopping_patience": 10,
    "use_augmentation": False,
}

# Transformer Model
TRANSFORMER_CONFIG = {
    # Hungarian BERT model
    "model_name": "SZTAKI-HLT/hubert-base-cc",
    "max_length": 128,
    "batch_size": 4,  # Smaller batch for better gradient updates
    "learning_rate": 0.00001,  # Lower LR for stability
    "num_epochs": 20,  # More epochs for small dataset
    "warmup_ratio": 0.1,  # Longer warmup
    "weight_decay": 0.01,
    "dropout": 0.4,  # Higher dropout to prevent overfitting
    "early_stopping_patience": 5,
    "gradient_accumulation_steps": 4,  # Effective batch size = 32
    "freeze_layers": 10,  # Freeze first 10 BERT layers
    "use_augmentation": False,  # Data augmentation DISABLED (not allowed)
    "label_smoothing": 0.0,  # Label smoothing for regularization
    "use_focal_loss": False,  # Focal loss for imbalanced data
    "use_augmentation": False,
}

# =============================================================================
# Training Settings
# =============================================================================
TRAIN_CONFIG = {
    "validation_split": 0.15,
    "test_split": 0.0,  # We have separate test data
    "stratify": True,
    "k_folds": 5,
    "shuffle": True,
}

# =============================================================================
# Evaluation Settings
# =============================================================================
EVAL_CONFIG = {
    "metrics": ["accuracy", "f1_macro", "f1_weighted", "mae", "confusion_matrix"],
    "consensus_threshold": 0.5,  # Agreement threshold for consensus labels
}

# =============================================================================
# Logging Settings
# =============================================================================
LOG_CONFIG = {
    "log_level": "INFO",
    "log_file": LOG_DIR / "run.log",
    "console_output": True,
}

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
