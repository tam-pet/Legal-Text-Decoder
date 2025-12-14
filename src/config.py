"""
Configuration file for Legal Text Decoder project.
Contains all hyperparameters, paths, and settings.

Structure:
- BASELINE: Logistic Regression (simple reference model)
- ADVANCED: XGBoost, RandomForest, GradientBoosting (to beat baseline)

Data characteristics (optimized for):
- Training: ~100 samples (very small - need strong regularization)
- Test: ~2500 samples (all individual annotations)
- Classes: 5 (ratings 1-5, imbalanced - Rating 1 & 5 underrepresented)
- Text length: avg ~50 words, max ~500 words (Hungarian legal text)
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

# Training data folder (single annotator)
TRAIN_FOLDER = "FA0B9B"
# Test data folder (all annotators - each annotation is a separate sample)
TEST_FOLDER = "consensus"


# =============================================================================
# General Settings
# =============================================================================
RANDOM_SEED = 42
NUM_CLASSES = 5  # Ratings 1-5 (legal text readability)


# =============================================================================
# TF-IDF Configuration (shared by all models)
# =============================================================================
# Optimized for small training set (~100 samples) with Hungarian legal text
TFIDF_CONFIG = {
    'max_features': 800,       # Reduced: prevent overfitting on small data
    'ngram_range': (1, 2),     # Unigram + bigram (trigram too sparse for 100 samples)
    'min_df': 2,               # At least 2 occurrences (filter noise)
    'max_df': 0.85,            # Remove very common words (stricter)
    'sublinear_tf': True,      # Apply log scaling (good for legal text)
}


# =============================================================================
# BASELINE MODEL: Logistic Regression
# =============================================================================
# Purpose: Simple reference model to establish baseline performance
# The advanced models should beat this baseline
BASELINE_CONFIG = {
    'name': 'Logistic Regression',
    'classifier': 'logistic_regression',
    'class_weight': 'balanced',    # Handle class imbalance (Rating 1&5 rare)
    'max_iter': 2000,              # More iterations for convergence
    'solver': 'lbfgs',
    'multi_class': 'multinomial',
    'C': 0.5,                      # Regularization strength (lower = more regularization)
    
    # TF-IDF settings (use shared config)
    'tfidf': TFIDF_CONFIG,
}


# =============================================================================
# ADVANCED MODELS 
# =============================================================================

# XGBoost - Gradient Boosting with regularization
# Optimized for small dataset with strong regularization
XGBOOST_CONFIG = {
    'name': 'XGBoost',
    'classifier': 'xgboost',
    
    # Tree parameters - conservative for small data
    'n_estimators': 80,            # Reduced: prevent overfitting
    'max_depth': 3,                # Shallow trees for small data
    'min_child_weight': 5,         # Higher: more conservative splits
    
    # Learning parameters
    'learning_rate': 0.08,         # Slightly higher with fewer trees
    'subsample': 0.7,              # More dropout for regularization
    'colsample_bytree': 0.7,       # Feature sampling
    
    # Regularization - strong for small dataset
    'gamma': 0.2,                  # Higher: more conservative
    'reg_alpha': 0.1,              # L1 regularization
    'reg_lambda': 2.0,             # L2 regularization (stronger)
    
    # Other
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'num_class': NUM_CLASSES,
    
    # TF-IDF settings
    'tfidf': TFIDF_CONFIG,
}


# RandomForest - Ensemble of decision trees
# Naturally handles small datasets well
RANDOMFOREST_CONFIG = {
    'name': 'Random Forest',
    'classifier': 'random_forest',
    
    # Tree parameters - optimized for small data
    'n_estimators': 150,           # More trees = more stable
    'max_depth': 5,                # Reduced: prevent overfitting
    'min_samples_split': 8,        # Higher: more conservative
    'min_samples_leaf': 4,         # Higher: smoother predictions
    'max_features': 'sqrt',        # Feature sampling
    
    # Other
    'class_weight': 'balanced',    # Handle imbalance
    'n_jobs': -1,
    'bootstrap': True,
    'oob_score': True,             # Out-of-bag score for validation
    
    # TF-IDF settings
    'tfidf': TFIDF_CONFIG,
}


# GradientBoosting - Sklearn's gradient boosting
# Good for small datasets with proper regularization
GRADIENTBOOSTING_CONFIG = {
    'name': 'Gradient Boosting',
    'classifier': 'gradient_boosting',
    
    # Tree parameters
    'n_estimators': 120,           # More trees with low learning rate
    'max_depth': 3,                # Shallow trees
    'min_samples_split': 8,        # Conservative splits
    'min_samples_leaf': 4,         # Smoother leaves
    'max_features': 'sqrt',        # Feature sampling
    
    # Learning parameters
    'learning_rate': 0.05,         # Higher than before (was too low)
    'subsample': 0.75,             # Stochastic gradient boosting
    
    # Validation
    'validation_fraction': 0.15,   # Early stopping validation
    'n_iter_no_change': 15,        # Early stopping patience
    
    # TF-IDF settings
    'tfidf': TFIDF_CONFIG,
}


# =============================================================================
# List of all models to train
# =============================================================================
# Change this to select which models to train
MODELS_TO_TRAIN = [
    'baseline',           # Always train baseline first
    'xgboost',            # Advanced model 1
    'random_forest',      # Advanced model 2
    'gradient_boosting',  # Advanced model 3
]


# =============================================================================
# Training Settings
# =============================================================================
TRAIN_CONFIG = {
    'validation_split': 0.15,      # ~15 samples for validation
    'test_split': 0.0,             # Separate test set (2591 samples)
    'stratify': True,              # Important for imbalanced classes
    'k_folds': 5,                  # 5-fold CV for reliable estimates
    'shuffle': True,
}


# =============================================================================
# Evaluation Settings
# =============================================================================
EVAL_CONFIG = {
    'metrics': ['accuracy', 'f1_macro', 'f1_weighted', 'mae', 'confusion_matrix'],
    'per_class_metrics': True,     # Detailed per-label analysis
}


# =============================================================================
# Logging Settings
# =============================================================================
LOG_CONFIG = {
    'log_level': 'INFO',
    'log_file': LOG_DIR / 'run.log',
    'console_output': True,
}


# =============================================================================
# Create directories if they don't exist
# =============================================================================
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
