"""
Training script for Legal Text Decoder project.
Implements both baseline and transformer-based models.
"""

import os
import sys
import json
import pickle
import re
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

# Deep learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Transformers
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, logging as transformers_logging
)
transformers_logging.set_verbosity_error()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR,
    BASELINE_CONFIG, MLP_CONFIG, TRANSFORMER_CONFIG, TRAIN_CONFIG,
    RANDOM_SEED, NUM_CLASSES
)
from utils import (
    setup_logger, log_config, log_model_summary, log_training_progress,
    set_seed, calculate_metrics, clean_text
)

# =============================================================================
# Simple MLP Model (Neural Network on TF-IDF features)
# =============================================================================

class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for TF-IDF features."""
    
    def __init__(self, input_size: int, hidden_layers: List[int], num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TFIDFDataset(Dataset):
    """PyTorch Dataset for TF-IDF features."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features.toarray() if hasattr(features, 'toarray') else features)
        self.labels = torch.LongTensor(labels - 1)  # Convert 1-5 to 0-4
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_mlp_model(train_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    config: Dict,
                    logger,
                    device: str = 'cuda') -> Tuple[Any, Dict[str, Any]]:
    """
    Train simple MLP on TF-IDF features.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        config: MLP configuration
        logger: Logger instance
        device: Device to use
        
    Returns:
        Trained model with vectorizer and metrics
    """
    logger.info("=" * 60)
    logger.info("TRAINING MLP MODEL (Simple Neural Network)")
    logger.info("=" * 60)
    
    log_config(logger, config, title="MLP Configuration")
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Prepare data
    X_train = train_df['text'].values
    y_train = train_df['label'].values
    X_val = val_df['text'].values
    y_val = val_df['label'].values
    
    # TF-IDF vectorization
    logger.info("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=config['tfidf_max_features'],
        ngram_range=config['tfidf_ngram_range'],
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    
    logger.info(f"TF-IDF shape: {X_train_tfidf.shape}")
    
    # Create datasets
    train_dataset = TFIDFDataset(X_train_tfidf, y_train)
    val_dataset = TFIDFDataset(X_val_tfidf, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    input_size = X_train_tfidf.shape[1]
    model = SimpleMLP(
        input_size=input_size,
        hidden_layers=config['hidden_layers'],
        num_classes=NUM_CLASSES,
        dropout=config['dropout']
    ).to(device)
    
    logger.info(f"MLP Architecture: {input_size} → {' → '.join(map(str, config['hidden_layers']))} → {NUM_CLASSES}")
    
    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    
    # Class weights
    class_counts = train_df['label'].value_counts().sort_index()
    class_weights = 1.0 / class_counts.values
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Training loop
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Training
        model.train()
        train_loss = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Convert back to 1-5 scale
        val_preds_adj = np.array(val_preds) + 1
        val_labels_adj = np.array(val_labels_list) + 1
        
        metrics = calculate_metrics(val_labels_adj, val_preds_adj)
        
        # Log progress
        if epoch % 5 == 0 or epoch == 1:
            log_training_progress(
                logger, epoch, config['num_epochs'],
                train_loss, {},
                val_loss, {'Acc': metrics['accuracy'], 'F1': metrics['f1_macro']}
            )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(metrics['accuracy'])
        history['val_f1'].append(metrics['f1_macro'])
        
        # Early stopping
        if metrics['f1_macro'] > best_val_f1:
            best_val_f1 = metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    logger.info(f"\nBest Validation F1: {best_val_f1:.4f}")
    logger.info(f"Final Validation Accuracy: {metrics['accuracy']:.4f}")
    
    # Package model with vectorizer
    model_package = {
        'model': model,
        'vectorizer': tfidf,
        'device': device,
        'config': config,
        'history': history
    }
    
    return model_package, metrics


# =============================================================================
# Focal Loss for Imbalanced Data
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices (0-4 for labels 1-5)
        """
        # Convert to probabilities
        p = torch.softmax(inputs, dim=1)
        
        # Get probability of true class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # ===== JAVÍTOTT: GAMMA CSÖKKENTVE =====
        # gamma=2.0 → túl agresszív, csökkentsd 1.5-re!
        focal_weight = (1 - p_t) ** 1.5  # ← JAVÍTVA: 2.0 → 1.5
        loss = focal_weight * ce_loss
        
        # Apply alpha weights
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



# =============================================================================
# Dataset Class
# =============================================================================

class LegalTextDataset(Dataset):
    """PyTorch Dataset for legal text classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label - 1, dtype=torch.long)  # Convert 1-5 to 0-4
        }

# =============================================================================
# Linguistic Feature Extractor
# =============================================================================


class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract linguistic features from text."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract features for each text."""
        from utils import extract_advanced_features
        
        features_list = []
        for text in X:
            features = extract_advanced_features(text)
            # Konvertáljuk lista formára (8 feature)
            features_list.append([
                features['word_count'],
                features['char_count'],
                features['avg_word_length'],
                features['sentence_count'],
                features['legal_word_count'],
                features['legal_word_ratio'],
                features['complex_word_ratio'],
                features['punctuation_ratio']
            ])
        
        return np.array(features_list)


# =============================================================================
# Baseline Models
# =============================================================================


def train_baseline_model(train_df: pd.DataFrame, 
                         val_df: pd.DataFrame,
                         config: Dict,
                         logger) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train baseline TF-IDF + classifier model.
    
    Args:
        train_df: Training dataframe with 'text' and 'label' columns
        val_df: Validation dataframe
        config: Baseline configuration
        logger: Logger instance
        
    Returns:
        Trained pipeline and metrics
    """
    logger.info("=" * 60)
    logger.info("TRAINING BASELINE MODEL")
    logger.info("=" * 60)
    
    log_config(logger, config, title="Baseline Configuration")
    
    
    # Prepare data
    X_train = train_df['text'].values
    y_train = train_df['label'].values
    X_val = val_df['text'].values
    y_val = val_df['label'].values
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=config['tfidf_max_features'],
        ngram_range=config['tfidf_ngram_range'],
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    # Create classifier based on config
    classifier_type = config.get('classifier', 'logistic_regression')
    logger.info(f"Creating classifier: {classifier_type}")
    
    # Flag to track if we need to adjust labels for XGBoost
    adjust_labels_for_xgboost = False
    
    if classifier_type == 'logistic_regression':
        clf = LogisticRegression(
            class_weight=config['class_weight'],
            max_iter=1000,
            random_state=RANDOM_SEED,
            multi_class='multinomial',
            solver='lbfgs'
        )
    elif classifier_type == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight=config['class_weight'],
            random_state=RANDOM_SEED,
            n_jobs=-1,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        )
    elif classifier_type == 'gradient_boosting':
        clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=RANDOM_SEED,
            verbose=0
        )
        logger.info("GradientBoostingClassifier parameters:")
        logger.info(f"  n_estimators: 200")
        logger.info(f"  learning_rate: 0.1")
        logger.info(f"  max_depth: 5")
    elif classifier_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        # XGBoost requires 0-indexed labels!
        adjust_labels_for_xgboost = True
        logger.info("Adjusting labels for XGBoost (1-5 → 0-4)")
        
        # Calculate scale_pos_weight for class imbalance
        class_counts = train_df['label'].value_counts()
        
        clf = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.01,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
            objective='multi:softmax',
            num_class=NUM_CLASSES,
            eval_metric='mlogloss'
        )
        logger.info("XGBoost parameters:")
        logger.info(f"  n_estimators: 200")
        logger.info(f"  learning_rate: 0.1")
        logger.info(f"  max_depth: 6")
        logger.info(f"  objective: multi:softmax")
        logger.info(f"  num_class: {NUM_CLASSES}")
    elif classifier_type == 'svm':
        clf = SVC(
            class_weight=config['class_weight'],
            random_state=RANDOM_SEED,
            kernel='rbf',
            probability=True,
            C=1.0
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}. Supported: logistic_regression, random_forest, gradient_boosting, xgboost, svm")
    
    # Adjust labels if using XGBoost
    if adjust_labels_for_xgboost:
        y_train_adjusted = y_train - 1  # Convert 1-5 to 0-4
        y_val_adjusted = y_val - 1
    else:
        y_train_adjusted = y_train
        y_val_adjusted = y_val
    
    # ========================================================================
    # ÚJ: 5-FOLD CROSS-VALIDATION (IDE ILLESZD BE!)
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("K-FOLD CROSS-VALIDATION (5 FOLDS)")
    logger.info("="*60)
    
    # Kombináld train + val-t teljes cross-validationhoz
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    X_full = full_df['text'].values
    y_full = full_df['label'].values
    
    # Adjust labels for full dataset if using XGBoost
    if adjust_labels_for_xgboost:
        y_full_adjusted = y_full - 1
    else:
        y_full_adjusted = y_full
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full_adjusted), 1):
        logger.info(f"\n--- FOLD {fold}/5 ---")
        
        X_train_fold = X_full[train_idx]
        y_train_fold = y_full_adjusted[train_idx]
        X_val_fold = X_full[val_idx]
        y_val_fold_adjusted = y_full_adjusted[val_idx]
        y_val_fold_original = y_full[val_idx]  # Keep original scale for metrics
        
        # Create fresh TF-IDF vectorizer for this fold
        tfidf_fold = TfidfVectorizer(
            max_features=config['tfidf_max_features'],
            ngram_range=config['tfidf_ngram_range'],
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Create fresh classifier (same type and params)
        if classifier_type == 'logistic_regression':
            clf_fold = LogisticRegression(
                class_weight=config['class_weight'],
                max_iter=1000,
                random_state=RANDOM_SEED,
                multi_class='multinomial',
                solver='lbfgs'
            )
        elif classifier_type == 'random_forest':
            clf_fold = RandomForestClassifier(
                n_estimators=200,
                class_weight=config['class_weight'],
                random_state=RANDOM_SEED,
                n_jobs=-1,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2
            )
        elif classifier_type == 'gradient_boosting':
            clf_fold = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=RANDOM_SEED,
                verbose=0
            )
        elif classifier_type == 'xgboost':
            clf_fold = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.01,
                reg_lambda=1.0,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=0,
                objective='multi:softmax',
                num_class=NUM_CLASSES,
                eval_metric='mlogloss'
            )
        elif classifier_type == 'svm':
            clf_fold = SVC(
                class_weight=config['class_weight'],
                random_state=RANDOM_SEED,
                kernel='rbf',
                probability=True,
                C=1.0
            )
        
        # Tanítás
        pipeline_fold = Pipeline([
            ('tfidf', tfidf_fold),
            ('classifier', clf_fold)
        ])
        pipeline_fold.fit(X_train_fold, y_train_fold)
        
        # Validáció
        y_pred_fold = pipeline_fold.predict(X_val_fold)
        
        # Convert back to 1-5 scale if using XGBoost
        if adjust_labels_for_xgboost:
            y_pred_fold_original = y_pred_fold + 1
        else:
            y_pred_fold_original = y_pred_fold
        
        metrics_fold = calculate_metrics(y_val_fold_original, y_pred_fold_original)
        fold_metrics.append(metrics_fold)
        
        logger.info(f"  Accuracy: {metrics_fold['accuracy']:.4f}")
        logger.info(f"  F1 Macro: {metrics_fold['f1_macro']:.4f}")
    
    # Átlag metrikák
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1_macro'] for m in fold_metrics])
    std_accuracy = np.std([m['accuracy'] for m in fold_metrics])
    std_f1 = np.std([m['f1_macro'] for m in fold_metrics])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CROSS-VALIDATION RESULTS:")
    logger.info(f"  Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    logger.info(f"  F1 Macro: {avg_f1:.4f} (±{std_f1:.4f})")
    logger.info(f"{'='*60}\n")
    # ========================================================================
    # VÉGE: K-FOLD CROSS-VALIDATION
    # ========================================================================
    
    # Create pipeline (EREDETI KÓD FOLYTATÓDIK)
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', clf)
    ])
    
    # Train
    logger.info("Training baseline model...")
    pipeline.fit(X_train, y_train_adjusted)
    
    # Log model info
    log_model_summary(logger, clf, "Baseline Model")
    
    n_features = len(pipeline.named_steps['tfidf'].vocabulary_)
    logger.info(f"TF-IDF features: {n_features}")
    
    # Cross-validation on training data
    logger.info("Performing cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train_adjusted, cv=5, scoring='accuracy')
    logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Validation evaluation
    y_pred = pipeline.predict(X_val)
    
    # Convert predictions back to 1-5 scale if using XGBoost
    if adjust_labels_for_xgboost:
        y_pred_original_scale = y_pred + 1
        y_val_original_scale = y_val  # Already in 1-5 scale
    else:
        y_pred_original_scale = y_pred
        y_val_original_scale = y_val
    
    metrics = calculate_metrics(y_val_original_scale, y_pred_original_scale)
    
    logger.info("\nValidation Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{metrics['confusion_matrix']}")
    
    # Store whether we adjusted labels in the pipeline for inference
    pipeline.label_adjustment = adjust_labels_for_xgboost
    
    return pipeline, metrics

# =============================================================================
# Transformer Model
# =============================================================================


class TransformerClassifier(nn.Module):
    """BERT-based classifier using AutoModelForSequenceClassification."""
    
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.3, freeze_layers: int = 0):
        super().__init__()
        
        # ===== JAVÍTVA: AutoModelForSequenceClassification =====
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        config.problem_type = "single_label_classification"
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
        # Freeze first N layers (BERT has 12 layers)
        if freeze_layers > 0:
            # Freeze embeddings
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
            
            # Freeze encoder layers (0 to freeze_layers-1)
            for i in range(freeze_layers):
                for param in self.model.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        # AutoModelForSequenceClassification returns SequenceClassifierOutput
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.logits  # Shape: (batch_size, num_classes)


def train_transformer_model(train_df: pd.DataFrame,
                            val_df: pd.DataFrame,
                            config: Dict,
                            logger,
                            device: str = 'cuda') -> Tuple[nn.Module, AutoTokenizer, Dict[str, Any]]:
    """
    Train transformer-based model.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        config: Transformer configuration
        logger: Logger instance
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Trained model, tokenizer, and metrics
    """
    logger.info("=" * 60)
    logger.info("TRAINING TRANSFORMER MODEL")
    logger.info("=" * 60)
    
    log_config(logger, config, title="Transformer Configuration")
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Apply data augmentation if enabled
    if config.get('use_augmentation', False):
        aug_factor = config.get('augmentation_factor', 2)
        logger.info(f"Applying data augmentation (factor={aug_factor})...")
        original_size = len(train_df)
        train_df = augment_dataset(train_df, factor=aug_factor)
        logger.info(f"Training data augmented: {original_size} -> {len(train_df)} samples")
    
    # Prepare datasets
    train_dataset = LegalTextDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        config['max_length']
    )
    
    val_dataset = LegalTextDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        config['max_length']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    freeze_layers = config.get('freeze_layers', 0)
    if freeze_layers > 0:
        logger.info(f"Freezing first {freeze_layers} BERT layers...")
    
    model = TransformerClassifier(
        config['model_name'],
        NUM_CLASSES,
        config['dropout'],
        freeze_layers=freeze_layers
    ).to(device)
    
    # Log model summary
    log_model_summary(logger, model, "Transformer Model")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * config['num_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function with class weights
    class_counts = train_df['label'].value_counts().sort_index()
    total = len(train_df)
    
    # ===== JAVÍTOTT: BALANCED CLASS WEIGHTS (INVERTED!) =====
    class_weights = []
    for label in range(1, NUM_CLASSES + 1):
        count = class_counts.get(label, 1)
        # INVERTED: Nagyobb osztály → NAGYOBB súly (fordított logika!)
        weight = count / total  # Label 4 (197/104) = 1.89 → LEGNAGYOBB!
        class_weights.append(weight)
    
    class_weights = np.array(class_weights, dtype=np.float32)
    
    # ===== ÚJ: NORMALIZE WEIGHTS (SUM = NUM_CLASSES) =====
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    
    logger.info(f"Class weights (inverted & normalized): {class_weights}")
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # ===== FONTOS: CROSS ENTROPY LOSS (NEM FOCAL!) =====
    # Focal Loss túl agresszív → Label 4-re ragad!
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info("Using Cross Entropy Loss with INVERTED class weights")
        
    # Label smoothing if configured
    label_smoothing = config.get('label_smoothing', 0.0)
    if label_smoothing > 0:
        logger.info(f"Using label smoothing: {label_smoothing}")
        if not config.get('use_focal_loss', False):
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    # Training loop
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Convert back to 1-5 scale for metrics
        val_preds_adj = np.array(val_preds) + 1
        val_labels_adj = np.array(val_labels) + 1
        
        metrics = calculate_metrics(val_labels_adj, val_preds_adj)
        
        # Log progress
        log_training_progress(
            logger, epoch, config['num_epochs'],
            train_loss, {'Acc': train_accuracy},
            val_loss, {'Acc': metrics['accuracy'], 'F1': metrics['f1_macro']}
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(metrics['accuracy'])
        history['val_f1'].append(metrics['f1_macro'])
        
        # Early stopping
        if metrics['f1_macro'] > best_val_f1:
            best_val_f1 = metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"  -> New best model! F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final validation
    model.eval()
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_preds_adj = np.array(val_preds) + 1
    val_labels_adj = np.array(val_labels) + 1
    
    final_metrics = calculate_metrics(val_labels_adj, val_preds_adj)
    
    logger.info("\nFinal Validation Results:")
    logger.info(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro: {final_metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Weighted: {final_metrics['f1_weighted']:.4f}")
    logger.info(f"  MAE: {final_metrics['mae']:.4f}")
    logger.info(f"  Cohen's Kappa: {final_metrics['cohen_kappa']:.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{final_metrics['confusion_matrix']}")
    
    return model, tokenizer, final_metrics, history


def save_models(baseline_pipeline: Optional[Pipeline],
                mlp_package: Optional[Dict],
                transformer_model: Optional[nn.Module],
                tokenizer: Optional[AutoTokenizer],
                transformer_config: Dict,
                output_dir: Path,
                logger) -> None:
    """Save trained models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save baseline model
    if baseline_pipeline:
        baseline_path = output_dir / "baseline_model.pkl"
        joblib.dump(baseline_pipeline, baseline_path)
        logger.info(f"Saved baseline model to {baseline_path}")
    
    # Save MLP model
    if mlp_package:
        mlp_path = output_dir / "mlp_model.pkl"
        joblib.dump(mlp_package, mlp_path)
        logger.info(f"Saved MLP model to {mlp_path}")
    
    # Save transformer model
    if transformer_model:
        transformer_dir = output_dir / "transformer"
        transformer_dir.mkdir(exist_ok=True)
        
        torch.save(transformer_model.state_dict(), transformer_dir / "model.pt")
        tokenizer.save_pretrained(transformer_dir)
        
        with open(transformer_dir / "config.json", 'w') as f:
            json.dump(transformer_config, f, indent=2)
        
        logger.info(f"Saved transformer model to {transformer_dir}")


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
    
    set_seed(RANDOM_SEED)
    
    # Load data
    logger.info("\nLoading processed data...")
    train_path = PROCESSED_DATA_DIR / "train.csv"
    
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        logger.error("Please run 01_data_preprocessing.py first")
        return
    
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df)} training samples")
    
    # Clean texts
    train_df['text'] = train_df['text'].apply(clean_text)
    train_df = train_df[train_df['text'].str.len() > 0]
    
    # Split into train/validation
    train_data, val_data = train_test_split(
        train_df,
        test_size=TRAIN_CONFIG['validation_split'],
        random_state=RANDOM_SEED,
        stratify=train_df['label'] if TRAIN_CONFIG['stratify'] else None
    )
    
    logger.info(f"Training set: {len(train_data)} samples")
    logger.info(f"Validation set: {len(val_data)} samples")
    
    # Train baseline model
    logger.info("\n" + "=" * 60)
    baseline_pipeline, baseline_metrics = train_baseline_model(
        train_data, val_data, BASELINE_CONFIG, logger
    )
    
    # Train MLP model (Simple Neural Network)
    logger.info("\n" + "=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlp_package, mlp_metrics = train_mlp_model(
        train_data, val_data, MLP_CONFIG, logger, device
    )
    
    # Train transformer model
    logger.info("\n" + "=" * 60)
    transformer_model, tokenizer, transformer_metrics, history = train_transformer_model(
        train_data, val_data, TRANSFORMER_CONFIG, logger, device
    )
    
    # Save models
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODELS")
    logger.info("=" * 60)
    
    save_models(
        baseline_pipeline,
        mlp_package,
        transformer_model,
        tokenizer,
        TRANSFORMER_CONFIG,
        MODEL_DIR,
        logger
    )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"\nBaseline Model (TF-IDF + {BASELINE_CONFIG['classifier'].upper()}):")
    logger.info(f"  Validation Accuracy: {baseline_metrics['accuracy']:.4f}")
    logger.info(f"  Validation F1 Macro: {baseline_metrics['f1_macro']:.4f}")
    
    logger.info(f"\nMLP Model (Simple Neural Network):")
    logger.info(f"  Validation Accuracy: {mlp_metrics['accuracy']:.4f}")
    logger.info(f"  Validation F1 Macro: {mlp_metrics['f1_macro']:.4f}")
    
    logger.info(f"\nTransformer Model ({TRANSFORMER_CONFIG['model_name']}):")
    logger.info(f"  Validation Accuracy: {transformer_metrics['accuracy']:.4f}")
    logger.info(f"  Validation F1 Macro: {transformer_metrics['f1_macro']:.4f}")
    
    # Save training history
    history_path = MODEL_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"\nSaved training history to {history_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
