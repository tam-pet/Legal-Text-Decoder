"""
Configuration file for Legal Text Decoder project.
Contains all hyperparameters, paths, and settings.

Structure:
- BASELINE: Logistic Regression (simple reference model)
- ADVANCED: XGBoost, RandomForest, GradientBoosting (to beat baseline)
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

# Training data folder
TRAIN_FOLDER = "FA0B9B"
# Test data folder
TEST_FOLDER = "consensus"


# =============================================================================
# General Settings
# =============================================================================
RANDOM_SEED = 42
NUM_CLASSES = 5  # Ratings 1-5 (legal text readability)


# =============================================================================
# TF-IDF Configuration (shared by all models)
# =============================================================================
TFIDF_CONFIG = {
    'max_features': 500,
    'ngram_range': (1, 2),     # Unigram + bigram 
    'min_df': 2,               # Keep rare words (small dataset)
    'max_df': 0.8,             # Remove very common words
    'sublinear_tf': True,      # Apply log scaling
}


# =============================================================================
# BASELINE MODEL: Logistic Regression
# =============================================================================
# Purpose: Simple reference model to establish baseline performance
# The advanced models should beat this baseline
BASELINE_CONFIG = {
    'name': 'Logistic Regression',
    'classifier': 'logistic_regression',
    'C': 0.1,
    'class_weight': 'balanced',    # Handle class imbalance
    'max_iter': 1000,
    'solver': 'lbfgs',
    'multi_class': 'multinomial',
    'penalty': 'l2',
    
    # TF-IDF settings (use shared config)
    'tfidf': TFIDF_CONFIG,
}


# =============================================================================
# ADVANCED MODELS 
# =============================================================================

# XGBoost - Gradient Boosting with regularization
XGBOOST_CONFIG = {
    'name': 'XGBoost',
    'classifier': 'xgboost',
    
    # Tree parameters
    'n_estimators': 50,
    'max_depth': 2,
    'min_child_weight': 8,
    
    # Learning parameters
    'learning_rate': 0.01,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    
    # Regularization
    'gamma': 0.5,
    'reg_alpha': 0.1,
    'reg_lambda': 5.0,
    
    # Other
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    
    # TF-IDF settings
    'tfidf': TFIDF_CONFIG,
}


# RandomForest - Ensemble of decision trees
RANDOMFOREST_CONFIG = {
    'name': 'Random Forest',
    'classifier': 'random_forest',
    
    # Tree parameters
    'n_estimators': 50,
    'max_depth': 3,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    
    # Other
    'class_weight': 'balanced',
    'n_jobs': -1,
    
    # TF-IDF settings
    'tfidf': TFIDF_CONFIG,
}


# GradientBoosting - Sklearn's gradient boosting
GRADIENTBOOSTING_CONFIG = {
    'name': 'Gradient Boosting',
    'classifier': 'gradient_boosting',
    
    # Tree parameters
    'n_estimators': 50,
    'max_depth': 2,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    
    # Learning parameters
    'learning_rate': 0.05,
    'subsample': 0.5,
    
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
    'validation_split': 0.15,
    'test_split': 0.0,           
    'stratify': True,
    'k_folds': 5,               
    'shuffle': True,
}


# =============================================================================
# Evaluation Settings
# =============================================================================
EVAL_CONFIG = {
    'metrics': ['accuracy', 'f1_macro', 'f1_weighted', 'mae', 'confusion_matrix'],
    'consensus_threshold': 0.5,  
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
