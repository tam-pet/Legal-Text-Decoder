"""
Training script for Legal Text Decoder project.
===============================================

This script trains multiple ML models for legal text readability classification.

Models:
- BASELINE: Logistic Regression (simple reference model)
- ADVANCED: XGBoost, RandomForest, GradientBoosting

All models use TF-IDF vectorization for feature extraction.
K-Fold cross-validation is used to get reliable performance estimates.

Usage:
    python src/a02_training.py
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import joblib

# XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, TRAIN_CONFIG,
    RANDOM_SEED, NUM_CLASSES, TFIDF_CONFIG,
    BASELINE_CONFIG, XGBOOST_CONFIG, RANDOMFOREST_CONFIG, 
    GRADIENTBOOSTING_CONFIG, MODELS_TO_TRAIN
)
from utils import (
    setup_logger, log_config, log_model_summary, log_training_progress,
    set_seed, calculate_metrics, clean_text
)


# =============================================================================
# Helper Functions
# =============================================================================

def create_tfidf_vectorizer(config: Dict) -> TfidfVectorizer:
    """
    Create TF-IDF vectorizer with given configuration.
    
    Args:
        config: TF-IDF configuration dictionary
        
    Returns:
        TfidfVectorizer instance
    """
    return TfidfVectorizer(
        max_features=config['max_features'],
        ngram_range=config['ngram_range'],
        min_df=config['min_df'],
        max_df=config['max_df'],
        sublinear_tf=config['sublinear_tf']
    )


def create_classifier(classifier_type: str, config: Dict):
    """
    Create classifier based on type and configuration.
    
    Args:
        classifier_type: Type of classifier ('logistic_regression', 'xgboost', etc.)
        config: Classifier configuration
        
    Returns:
        Classifier instance
    """
    if classifier_type == 'logistic_regression':
        return LogisticRegression(
            class_weight=config.get('class_weight', 'balanced'),
            max_iter=config.get('max_iter', 1000),
            solver=config.get('solver', 'lbfgs'),
            multi_class=config.get('multi_class', 'multinomial'),
            random_state=RANDOM_SEED
        )
    
    elif classifier_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        return xgb.XGBClassifier(
            n_estimators=config.get('n_estimators', 200),
            max_depth=config.get('max_depth', 6),
            min_child_weight=config.get('min_child_weight', 3),
            learning_rate=config.get('learning_rate', 0.05),
            subsample=config.get('subsample', 0.8),
            colsample_bytree=config.get('colsample_bytree', 0.8),
            gamma=config.get('gamma', 0.1),
            reg_alpha=config.get('reg_alpha', 0.01),
            reg_lambda=config.get('reg_lambda', 1.0),
            objective='multi:softmax',
            num_class=NUM_CLASSES,
            eval_metric='mlogloss',
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0
        )
    
    elif classifier_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=config.get('n_estimators', 200),
            max_depth=config.get('max_depth', 20),
            min_samples_split=config.get('min_samples_split', 5),
            min_samples_leaf=config.get('min_samples_leaf', 2),
            class_weight=config.get('class_weight', 'balanced'),
            random_state=RANDOM_SEED,
            n_jobs=config.get('n_jobs', -1)
        )
    
    elif classifier_type == 'gradient_boosting':
        return GradientBoostingClassifier(
            n_estimators=config.get('n_estimators', 200),
            max_depth=config.get('max_depth', 5),
            min_samples_split=config.get('min_samples_split', 5),
            min_samples_leaf=config.get('min_samples_leaf', 2),
            learning_rate=config.get('learning_rate', 0.1),
            subsample=config.get('subsample', 0.8),
            random_state=RANDOM_SEED,
            verbose=0
        )

    
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")


def get_config_for_model(model_name: str) -> Dict:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model ('baseline', 'xgboost', etc.)
        
    Returns:
        Configuration dictionary
    """
    config_map = {
        'baseline': BASELINE_CONFIG,
        'xgboost': XGBOOST_CONFIG,
        'random_forest': RANDOMFOREST_CONFIG,
        'gradient_boosting': GRADIENTBOOSTING_CONFIG,
    }
    
    if model_name not in config_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(config_map.keys())}")
    
    return config_map[model_name]


# =============================================================================
# K-Fold Cross-Validation
# =============================================================================

def run_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    classifier_type: str,
    config: Dict,
    logger,
    n_folds: int = 5,
    use_xgb_labels: bool = False
) -> Dict[str, float]:
    """
    Run K-Fold cross-validation for a model.
    
    XGBoost requires 0-indexed labels (0-4), other models use 1-indexed (1-5).
    
    Args:
        X: Feature array (texts)
        y: Label array (1-5 scale)
        classifier_type: Type of classifier
        config: Model configuration
        logger: Logger instance
        n_folds: Number of folds
        use_xgb_labels: Whether to convert labels for XGBoost
        
    Returns:
        Dictionary with CV metrics (accuracy, f1_macro, std)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"K-FOLD CROSS-VALIDATION ({n_folds} FOLDS)")
    logger.info(f"{'='*60}")
    
    tfidf_config = config.get('tfidf', TFIDF_CONFIG)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    # Adjust labels for XGBoost if needed
    y_adjusted = y - 1 if use_xgb_labels else y
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_adjusted), 1):
        logger.info(f"\n--- FOLD {fold}/{n_folds} ---")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold = y_adjusted[train_idx]
        y_val_fold_adjusted = y_adjusted[val_idx]
        y_val_fold_original = y[val_idx]  # Keep original scale for metrics
        
        # Create fresh vectorizer and classifier for this fold
        tfidf_fold = create_tfidf_vectorizer(tfidf_config)
        clf_fold = create_classifier(classifier_type, config)
        
        # Create and train pipeline
        pipeline_fold = Pipeline([
            ('tfidf', tfidf_fold),
            ('classifier', clf_fold)
        ])
        pipeline_fold.fit(X_train_fold, y_train_fold)
        
        # Predict
        y_pred_fold = pipeline_fold.predict(X_val_fold)
        
        # Convert predictions back to 1-5 scale if using XGBoost
        if use_xgb_labels:
            y_pred_fold_original = y_pred_fold + 1
        else:
            y_pred_fold_original = y_pred_fold
        
        # Calculate metrics
        metrics_fold = calculate_metrics(y_val_fold_original, y_pred_fold_original)
        fold_metrics.append(metrics_fold)
        
        logger.info(f"  Accuracy: {metrics_fold['accuracy']:.4f}")
        logger.info(f"  F1 Macro: {metrics_fold['f1_macro']:.4f}")
    
    # Calculate average metrics
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1_macro'] for m in fold_metrics])
    std_accuracy = np.std([m['accuracy'] for m in fold_metrics])
    std_f1 = np.std([m['f1_macro'] for m in fold_metrics])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CROSS-VALIDATION RESULTS:")
    logger.info(f"  Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    logger.info(f"  F1 Macro: {avg_f1:.4f} (±{std_f1:.4f})")
    logger.info(f"{'='*60}\n")
    
    return {
        'cv_accuracy': avg_accuracy,
        'cv_accuracy_std': std_accuracy,
        'cv_f1_macro': avg_f1,
        'cv_f1_macro_std': std_f1,
        'fold_metrics': fold_metrics
    }


# =============================================================================
# Model Training Function
# =============================================================================

def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    logger
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a single model with the given configuration.
    
    Args:
        train_df: Training dataframe with 'text' and 'label' columns
        val_df: Validation dataframe
        model_name: Name of model to train ('baseline', 'xgboost', etc.)
        logger: Logger instance
        
    Returns:
        Tuple of (trained pipeline, metrics dictionary)
    """
    # Get configuration
    config = get_config_for_model(model_name)
    classifier_type = config['classifier']
    tfidf_config = config.get('tfidf', TFIDF_CONFIG)
    
    # Determine if XGBoost (needs 0-indexed labels)
    use_xgb_labels = classifier_type == 'xgboost'
    
    logger.info("=" * 60)
    logger.info(f"TRAINING MODEL: {config['name']}")
    logger.info("=" * 60)
    
    log_config(logger, config, title=f"{config['name']} Configuration")
    
    # Prepare data
    X_train = train_df['text'].values
    y_train = train_df['label'].values
    X_val = val_df['text'].values
    y_val = val_df['label'].values
    
    logger.info(f"\nTraining samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    # Log label distribution
    logger.info("\nTraining label distribution:")
    for label in sorted(np.unique(y_train)):
        count = (y_train == label).sum()
        pct = count / len(y_train) * 100
        logger.info(f"  Label {label}: {count} ({pct:.1f}%)")
    
    # =========================================================================
    # Run K-Fold Cross-Validation
    # =========================================================================
    # Combine train + val for CV (to use all available data)
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    X_full = full_df['text'].values
    y_full = full_df['label'].values
    
    cv_results = run_kfold_cv(
        X_full, y_full, 
        classifier_type, config, 
        logger, 
        n_folds=TRAIN_CONFIG['k_folds'],
        use_xgb_labels=use_xgb_labels
    )
    
    # =========================================================================
    # Train Final Model
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING FINAL MODEL")
    logger.info("=" * 60)
    
    # Create vectorizer and classifier
    tfidf = create_tfidf_vectorizer(tfidf_config)
    clf = create_classifier(classifier_type, config)
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', clf)
    ])
    
    # Adjust labels for XGBoost if needed
    y_train_adjusted = y_train - 1 if use_xgb_labels else y_train
    y_val_adjusted = y_val - 1 if use_xgb_labels else y_val
    
    # Train
    logger.info(f"Training {config['name']}...")
    pipeline.fit(X_train, y_train_adjusted)
    
    # Log model info
    log_model_summary(logger, clf, config['name'])
    n_features = len(pipeline.named_steps['tfidf'].vocabulary_)
    logger.info(f"TF-IDF features: {n_features}")
    
    # =========================================================================
    # Evaluate on Validation Set
    # =========================================================================
    y_pred = pipeline.predict(X_val)
    
    # Convert predictions back to 1-5 scale if using XGBoost
    if use_xgb_labels:
        y_pred_original = y_pred + 1
    else:
        y_pred_original = y_pred
    
    # Calculate metrics
    metrics = calculate_metrics(y_val, y_pred_original)
    
    # Add CV results to metrics
    metrics.update(cv_results)
    
    logger.info("\nValidation Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{metrics['confusion_matrix']}")
    
    # Store label adjustment info in pipeline for inference
    pipeline.use_xgb_labels = use_xgb_labels
    pipeline.model_name = model_name
    pipeline.display_name = config['name']
    
    return pipeline, metrics


# =============================================================================
# Save Models
# =============================================================================

def save_model(
    pipeline: Pipeline,
    model_name: str,
    metrics: Dict,
    output_dir: Path,
    logger
) -> None:
    """
    Save trained model to disk.
    
    Args:
        pipeline: Trained sklearn pipeline
        model_name: Name of the model
        metrics: Training metrics
        output_dir: Directory to save model
        logger: Logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"{model_name}_model.pkl"
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved {model_name} model to {model_path}")
    
    # Save metrics
    metrics_path = output_dir / f"{model_name}_metrics.json"
    
    # Convert numpy types to Python types for JSON
    metrics_clean = {}
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            metrics_clean[key] = value.tolist() if hasattr(value, 'tolist') else value
        elif key == 'fold_metrics':
            # Skip fold metrics for JSON (too verbose)
            continue
        elif isinstance(value, np.ndarray):
            metrics_clean[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            metrics_clean[key] = float(value)
        else:
            metrics_clean[key] = value
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    logger.info(f"Saved {model_name} metrics to {metrics_path}")


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():
    """Main training pipeline."""
    # Setup logger
    logger = setup_logger(
        name="Training",
        log_file=LOG_DIR / "run.log",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("LEGAL TEXT DECODER - MODEL TRAINING")
    logger.info("=" * 60)
    logger.info("\nModel Architecture:")
    logger.info("  - BASELINE: Logistic Regression (reference model)")
    logger.info("  - ADVANCED: XGBoost, RandomForest, GradientBoosting")
    logger.info("  - All models use TF-IDF vectorization")
    
    set_seed(RANDOM_SEED)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    
    train_path = PROCESSED_DATA_DIR / "train.csv"
    
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        logger.error("Please run a01_data_preprocessing.py first")
        return
    
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df)} training samples")
        
    
    # Clean texts
    train_df['text'] = train_df['text'].apply(clean_text)
    train_df = train_df[train_df['text'].str.len() > 0]
    
    # Log label distribution
    logger.info("\nLabel distribution:")
    for label in sorted(train_df['label'].unique()):
        count = (train_df['label'] == label).sum()
        pct = count / len(train_df) * 100
        logger.info(f"  Label {label}: {count} ({pct:.1f}%)")
    
    # =========================================================================
    # Split into Train/Validation
    # =========================================================================
    train_data, val_data = train_test_split(
        train_df,
        test_size=TRAIN_CONFIG['validation_split'],
        random_state=RANDOM_SEED,
        stratify=train_df['label'] if TRAIN_CONFIG['stratify'] else None
    )
    
    logger.info(f"\nTraining set: {len(train_data)} samples")
    logger.info(f"Validation set: {len(val_data)} samples")
    
    # =========================================================================
    # Train All Models
    # =========================================================================
    trained_models = {}
    all_metrics = {}
    
    for model_name in MODELS_TO_TRAIN:
        logger.info("\n\n" + "=" * 60)
        logger.info(f">>> TRAINING: {model_name.upper()}")
        logger.info("=" * 60)
        
        try:
            pipeline, metrics = train_model(
                train_data, val_data, model_name, logger
            )
            trained_models[model_name] = pipeline
            all_metrics[model_name] = metrics
            
            # Save model
            save_model(pipeline, model_name, metrics, MODEL_DIR, logger)
            
        except ImportError as e:
            logger.warning(f"Skipping {model_name}: {e}")
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            raise
    
    # =========================================================================
    # Training Summary
    # =========================================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    # Create comparison table
    logger.info(f"\n{'Model':<25} {'CV Acc':<12} {'CV F1':<12} {'Val Acc':<12} {'Val F1':<12}")
    logger.info("-" * 75)
    
    for model_name, metrics in all_metrics.items():
        config = get_config_for_model(model_name)
        cv_acc = metrics.get('cv_accuracy', 0)
        cv_acc_std = metrics.get('cv_accuracy_std', 0)
        cv_f1 = metrics.get('cv_f1_macro', 0)
        cv_f1_std = metrics.get('cv_f1_macro_std', 0)
        val_acc = metrics['accuracy']
        val_f1 = metrics['f1_macro']
        
        logger.info(
            f"{config['name']:<25} "
            f"{cv_acc:.3f}±{cv_acc_std:.3f}  "
            f"{cv_f1:.3f}±{cv_f1_std:.3f}  "
            f"{val_acc:<12.4f} "
            f"{val_f1:<12.4f}"
        )
    
    # Find best model
    best_model = max(all_metrics.items(), key=lambda x: x[1]['cv_accuracy'])
    logger.info(f"\nBest model (by CV Accuracy): {best_model[0]} ({best_model[1]['cv_accuracy']:.4f})")
    
    # Save overall summary
    summary_path = MODEL_DIR / "training_summary.json"
    summary = {
        model_name: {
            'cv_accuracy': float(m.get('cv_accuracy', 0)),
            'cv_accuracy_std': float(m.get('cv_accuracy_std', 0)),
            'cv_f1_macro': float(m.get('cv_f1_macro', 0)),
            'cv_f1_macro_std': float(m.get('cv_f1_macro_std', 0)),
            'val_accuracy': float(m['accuracy']),
            'val_f1_macro': float(m['f1_macro']),
            'val_mae': float(m['mae']),
            'val_cohen_kappa': float(m['cohen_kappa'])
        }
        for model_name, m in all_metrics.items()
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved training summary to {summary_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
