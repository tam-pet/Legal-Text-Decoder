"""
Evaluation script for Legal Text Decoder project.
================================================

This script evaluates all trained models on the test set (consensus labels).

Features:
- Per-model evaluation with detailed metrics
- Per-sample analysis (correct/incorrect predictions)
- Per-label precision/recall/F1
- Model comparison summary
- Confusion matrices and visualizations

Usage:
    python src/a03_evaluation.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, mean_absolute_error,
    cohen_kappa_score, precision_recall_fscore_support
)
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, EVAL_CONFIG,
    NUM_CLASSES, MODELS_TO_TRAIN
)
from utils import (
    setup_logger, log_config, log_evaluation_results,
    set_seed, calculate_metrics, clean_text
)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_name: str, model_dir: Path, logger) -> Optional[Any]:
    """
    Load a trained model from disk.
    
    Args:
        model_name: Name of the model ('baseline', 'xgboost', etc.)
        model_dir: Directory containing model files
        logger: Logger instance
        
    Returns:
        Loaded model pipeline or None if not found
    """
    model_path = model_dir / f"{model_name}_model.pkl"
    
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None
    
    logger.info(f"Loading {model_name} model from {model_path}")
    return joblib.load(model_path)


def load_all_models(model_dir: Path, logger) -> Dict[str, Any]:
    """
    Load all available trained models.
    
    Args:
        model_dir: Directory containing model files
        logger: Logger instance
        
    Returns:
        Dictionary of model_name -> model
    """
    models = {}
    
    for model_name in MODELS_TO_TRAIN:
        model = load_model(model_name, model_dir, logger)
        if model is not None:
            models[model_name] = model
    
    logger.info(f"\nLoaded {len(models)} models: {list(models.keys())}")
    return models


# =============================================================================
# Model Evaluation
# =============================================================================

def evaluate_model(
    model,
    model_name: str,
    test_df: pd.DataFrame,
    logger
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluate a single model on the test set.
    
    Args:
        model: Trained sklearn pipeline
        model_name: Name of the model
        test_df: Test dataframe with 'text' and 'label' columns
        logger: Logger instance
        
    Returns:
        Tuple of (metrics dictionary, per-sample results dataframe)
    """
    display_name = getattr(model, 'display_name', model_name)
    use_xgb_labels = getattr(model, 'use_xgb_labels', False)
    
    logger.info("=" * 60)
    logger.info(f"EVALUATING: {display_name}")
    logger.info("=" * 60)
    
    X_test = test_df['text'].values
    y_test = test_df['label'].values
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Convert predictions back to 1-5 scale if using XGBoost
    if use_xgb_labels:
        y_pred = y_pred + 1
    
    # Get probabilities if available
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            pass
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Log main results
    log_evaluation_results(logger, {
        'Accuracy': metrics['accuracy'],
        'F1 Macro': metrics['f1_macro'],
        'F1 Weighted': metrics['f1_weighted'],
        'Precision Macro': metrics['precision_macro'],
        'Recall Macro': metrics['recall_macro'],
        'MAE': metrics['mae'],
        "Cohen's Kappa": metrics['cohen_kappa'],
    }, title=f"{display_name} - Test Results")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{metrics['confusion_matrix']}")
    
    logger.info("\nClassification Report:")
    logger.info(f"\n{metrics['classification_report']}")
    
    # Per-sample analysis
    results_df = analyze_per_sample(test_df, y_pred, y_proba, display_name, logger)
    
    return metrics, results_df


# =============================================================================
# Per-Sample Analysis
# =============================================================================

def analyze_per_sample(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    model_name: str,
    logger
) -> pd.DataFrame:
    """
    Analyze model performance for each individual sample.
    
    Args:
        test_df: Test dataframe
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        model_name: Name of the model
        logger: Logger instance
        
    Returns:
        DataFrame with per-sample results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PER-SAMPLE ANALYSIS: {model_name}")
    logger.info(f"{'='*60}")
    
    # Create detailed per-sample dataframe
    results_df = test_df.copy()
    results_df['predicted'] = y_pred
    results_df['error'] = np.abs(results_df['label'] - results_df['predicted'])
    results_df['correct'] = results_df['label'] == results_df['predicted']
    
    # Add prediction confidence if available
    if y_proba is not None:
        results_df['confidence'] = np.max(y_proba, axis=1)
        results_df['predicted_proba_true_class'] = [
            y_proba[i, label-1] for i, label in enumerate(results_df['label'])
        ]
    
    # Overall statistics
    logger.info(f"\nTotal samples: {len(results_df)}")
    logger.info(f"Correct predictions: {results_df['correct'].sum()} ({results_df['correct'].mean()*100:.1f}%)")
    logger.info(f"Average error: {results_df['error'].mean():.3f}")
    logger.info(f"Max error: {results_df['error'].max():.0f}")
    
    # Error distribution
    logger.info(f"\nError Distribution:")
    for error_val in sorted(results_df['error'].unique()):
        count = (results_df['error'] == error_val).sum()
        pct = count / len(results_df) * 100
        logger.info(f"  Error {int(error_val)}: {count} samples ({pct:.1f}%)")
    
    # Analyze by true label
    logger.info(f"\nAccuracy by True Label:")
    for label in sorted(results_df['label'].unique()):
        subset = results_df[results_df['label'] == label]
        accuracy = subset['correct'].mean()
        logger.info(f"  Label {label}: {accuracy*100:.1f}% ({subset['correct'].sum()}/{len(subset)})")
    
    # Most common confusions
    logger.info(f"\nMost Common Confusions (True -> Predicted):")
    confusions = results_df[~results_df['correct']].groupby(['label', 'predicted']).size()
    confusions = confusions.sort_values(ascending=False).head(10)
    for (true_label, pred_label), count in confusions.items():
        logger.info(f"  {true_label} -> {pred_label}: {count} times")
    
    # Per-label precision/recall/F1
    log_per_label_metrics(results_df, model_name, logger)
    
    # Error analysis by true label
    log_error_analysis(results_df, logger)
    
    # Hardest samples
    log_hardest_samples(results_df, y_proba, logger)
    
    # Confident mistakes
    if y_proba is not None:
        log_confident_mistakes(results_df, logger)
    
    return results_df


def log_per_label_metrics(results_df: pd.DataFrame, model_name: str, logger) -> None:
    """Log precision/recall/F1 for each label."""
    logger.info(f"\n{'='*60}")
    logger.info(f"PER-LABEL METRICS: {model_name}")
    logger.info(f"{'='*60}")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        results_df['label'], 
        results_df['predicted'], 
        labels=[1, 2, 3, 4, 5],
        zero_division=0
    )
    
    # Table header
    logger.info("")
    logger.info(f"{'Label':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    logger.info("-" * 60)
    
    # Rows
    for i, label in enumerate([1, 2, 3, 4, 5]):
        logger.info(
            f"{label:<8} "
            f"{precision[i]:<12.3f} "
            f"{recall[i]:<12.3f} "
            f"{f1[i]:<12.3f} "
            f"{support[i]:<10}"
        )
    
    # Averages
    logger.info("")
    logger.info(
        f"{'Macro Avg':<8} "
        f"{np.mean(precision):<12.3f} "
        f"{np.mean(recall):<12.3f} "
        f"{np.mean(f1):<12.3f}"
    )
    logger.info(
        f"{'Weighted':<8} "
        f"{np.average(precision, weights=support):<12.3f} "
        f"{np.average(recall, weights=support):<12.3f} "
        f"{np.average(f1, weights=support):<12.3f}"
    )


def log_error_analysis(results_df: pd.DataFrame, logger) -> None:
    """Log error analysis by true label."""
    logger.info(f"\n{'='*60}")
    logger.info(f"ERROR ANALYSIS BY TRUE LABEL")
    logger.info(f"{'='*60}")
    
    for label in sorted(results_df['label'].unique()):
        subset = results_df[results_df['label'] == label]
        errors = subset[~subset['correct']]
        
        logger.info(f"\nTrue Label = {label} ({len(subset)} samples):")
        logger.info(f"  Correct: {subset['correct'].sum()} ({subset['correct'].mean()*100:.1f}%)")
        logger.info(f"  Avg Error: {subset['error'].mean():.2f}")
        
        if len(errors) > 0:
            logger.info(f"  Most Common Mistakes:")
            for pred_label, count in errors['predicted'].value_counts().head(3).items():
                logger.info(f"    Predicted as {pred_label}: {count} times")


def log_hardest_samples(
    results_df: pd.DataFrame,
    y_proba: Optional[np.ndarray],
    logger,
    top_n: int = 10
) -> None:
    """Log the hardest samples (largest errors)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TOP {top_n} HARDEST SAMPLES (Largest Errors)")
    logger.info(f"{'='*60}")
    
    hardest = results_df.nlargest(top_n, 'error')
    
    for idx, row in hardest.iterrows():
        text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
        conf_str = f", Conf: {row['confidence']:.3f}" if y_proba is not None else ""
        logger.info(
            f"{idx}. True={row['label']}, Pred={row['predicted']}, "
            f"Error={row['error']}{conf_str} | {text_preview}"
        )


def log_confident_mistakes(results_df: pd.DataFrame, logger, top_n: int = 5) -> None:
    """Log confident mistakes (high confidence, wrong prediction)."""
    if 'confidence' not in results_df.columns:
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TOP {top_n} CONFIDENT MISTAKES (High Confidence, Wrong)")
    logger.info(f"{'='*60}")
    
    mistakes = results_df[~results_df['correct']]
    if len(mistakes) == 0:
        logger.info("No mistakes found!")
        return
    
    confident_mistakes = mistakes.nlargest(top_n, 'confidence')
    
    for idx, row in confident_mistakes.iterrows():
        text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
        logger.info(
            f"{idx}. True={row['label']}, Pred={row['predicted']}, "
            f"Conf={row['confidence']:.3f} | {text_preview}"
        )


# =============================================================================
# Visualization
# =============================================================================

def plot_confusion_matrices(
    all_metrics: Dict[str, Dict],
    output_dir: Path,
    logger
) -> None:
    """Plot confusion matrices for all models."""
    n_models = len(all_metrics)
    if n_models == 0:
        return
    
    # Calculate grid size
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (model_name, metrics) in enumerate(all_metrics.items()):
        if 'confusion_matrix' not in metrics:
            continue
        
        cm = metrics['confusion_matrix']
        if isinstance(cm, list):
            cm = np.array(cm)
        
        sns.heatmap(
            cm,
            annot=True, fmt='d', cmap='Blues',
            xticklabels=range(1, 6), yticklabels=range(1, 6),
            ax=axes[idx]
        )
        axes[idx].set_title(f'{model_name.replace("_", " ").title()}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    output_path = output_dir / "confusion_matrices.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved confusion matrices to {output_path}")
    plt.close()


def plot_model_comparison(
    all_metrics: Dict[str, Dict],
    output_dir: Path,
    logger
) -> None:
    """Plot model comparison bar chart."""
    metrics_to_compare = ['accuracy', 'f1_macro', 'f1_weighted', 'cohen_kappa']
    labels = ['Accuracy', 'F1 Macro', 'F1 Weighted', "Cohen's Kappa"]
    
    x = np.arange(len(labels))
    width = 0.8 / len(all_metrics)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_metrics)))
    
    for idx, (model_name, metrics) in enumerate(all_metrics.items()):
        values = [metrics.get(m, 0) for m in metrics_to_compare]
        offset = (idx - len(all_metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name.replace('_', ' ').title(), color=colors[idx])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison on Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    output_path = output_dir / "model_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved model comparison to {output_path}")
    plt.close()


# =============================================================================
# Save Results
# =============================================================================

def save_results(
    all_metrics: Dict[str, Dict],
    output_dir: Path,
    logger
) -> None:
    """Save evaluation results to JSON."""
    results = {}
    
    for model_name, metrics in all_metrics.items():
        results[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_weighted': float(metrics['f1_weighted']),
            'mae': float(metrics['mae']),
            'cohen_kappa': float(metrics['cohen_kappa']),
            'precision_macro': float(metrics.get('precision_macro', 0)),
            'recall_macro': float(metrics.get('recall_macro', 0)),
        }
    
    output_path = output_dir / "evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation results to {output_path}")


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def main():
    """Main evaluation pipeline."""
    # Setup logger
    logger = setup_logger(
        name="Evaluation",
        log_file=LOG_DIR / "run.log",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("LEGAL TEXT DECODER - MODEL EVALUATION")
    logger.info("=" * 60)
    
    # =========================================================================
    # Load Test Data
    # =========================================================================
    logger.info("\nLoading test data...")
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    if not test_path.exists():
        logger.error(f"Test data not found at {test_path}")
        logger.error("Please run a01_data_preprocessing.py first")
        return
    
    test_df = pd.read_csv(test_path)
    test_df['text'] = test_df['text'].apply(clean_text)
    test_df = test_df[test_df['text'].str.len() > 0]
    
    logger.info(f"Loaded {len(test_df)} test samples")
    
    # Log test data distribution
    logger.info("\nTest label distribution:")
    for label in sorted(test_df['label'].unique()):
        count = (test_df['label'] == label).sum()
        pct = count / len(test_df) * 100
        logger.info(f"  Rating {label}: {count} ({pct:.1f}%)")
    
    # =========================================================================
    # Load Models
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("LOADING MODELS")
    logger.info("=" * 60)
    
    models = load_all_models(MODEL_DIR, logger)
    
    if not models:
        logger.error("No models found! Please run a02_training.py first")
        return
    
    # =========================================================================
    # Evaluate All Models
    # =========================================================================
    all_metrics = {}
    all_results = {}
    
    for model_name, model in models.items():
        logger.info("\n\n")
        metrics, results_df = evaluate_model(model, model_name, test_df, logger)
        all_metrics[model_name] = metrics
        all_results[model_name] = results_df
    
    # =========================================================================
    # Model Comparison Summary
    # =========================================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    # Create comparison table
    logger.info(f"\n{'Model':<20} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'MAE':>8} {'Kappa':>8}")
    logger.info("-" * 75)
    
    for model_name, metrics in all_metrics.items():
        display_name = model_name.replace('_', ' ').title()
        logger.info(
            f"{display_name:<20} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['f1_macro']:>10.4f} "
            f"{metrics['f1_weighted']:>12.4f} "
            f"{metrics['mae']:>8.4f} "
            f"{metrics['cohen_kappa']:>8.4f}"
        )
    
    # Find best model
    best_model = max(all_metrics.items(), key=lambda x: x[1]['f1_macro'])
    logger.info(f"\nBest model (by F1 Macro): {best_model[0]} ({best_model[1]['f1_macro']:.4f})")
    
    # Check if any advanced model beats baseline
    if 'baseline' in all_metrics:
        baseline_f1 = all_metrics['baseline']['f1_macro']
        for model_name, metrics in all_metrics.items():
            if model_name != 'baseline':
                diff = metrics['f1_macro'] - baseline_f1
                status = "[+] BEATS BASELINE" if diff > 0 else "[-] Below baseline"
                logger.info(f"  {model_name}: {status} (diff: {diff:+.4f})")
    
    # =========================================================================
    # Generate Visualizations
    # =========================================================================
    logger.info("\n\nGenerating evaluation plots...")
    output_dir = MODEL_DIR / "evaluation"
    output_dir.mkdir(exist_ok=True)
    
    plot_confusion_matrices(all_metrics, output_dir, logger)
    plot_model_comparison(all_metrics, output_dir, logger)
    
    # =========================================================================
    # Save Results
    # =========================================================================
    save_results(all_metrics, output_dir, logger)
    
    # Save per-sample results for each model
    for model_name, results_df in all_results.items():
        csv_path = output_dir / f"{model_name}_per_sample_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {model_name} per-sample results to {csv_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
