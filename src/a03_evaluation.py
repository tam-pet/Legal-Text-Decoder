"""
Evaluation script for Legal Text Decoder project.
Evaluates trained models on the test set with consensus labels.
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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, mean_absolute_error,
    cohen_kappa_score
)
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, EVAL_CONFIG,
    TRANSFORMER_CONFIG, NUM_CLASSES
)
from utils import (
    setup_logger, log_config, log_evaluation_results,
    set_seed, calculate_metrics, clean_text
)
from a02_training import LegalTextDataset, TransformerClassifier


def load_baseline_model(model_dir: Path, logger):
    """Load baseline model from disk."""
    model_path = model_dir / "baseline_model.pkl"
    
    if not model_path.exists():
        logger.warning(f"Baseline model not found at {model_path}")
        return None
    
    logger.info(f"Loading baseline model from {model_path}")
    return joblib.load(model_path)


def load_transformer_model(model_dir: Path, device: str, logger):
    """Load transformer model from disk."""
    transformer_dir = model_dir / "transformer"
    
    if not transformer_dir.exists():
        logger.warning(f"Transformer model not found at {transformer_dir}")
        return None, None
    
    logger.info(f"Loading transformer model from {transformer_dir}")
    
    # Load config
    config_path = transformer_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
    
    # Create and load model
    model = TransformerClassifier(
        config['model_name'],
        NUM_CLASSES,
        config.get('dropout', 0.1)
    )
    
    model.load_state_dict(torch.load(transformer_dir / "model.pt", map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def evaluate_baseline(model, test_df: pd.DataFrame, logger) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate baseline model on test data."""
    logger.info("=" * 60)
    logger.info("EVALUATING BASELINE MODEL")
    logger.info("=" * 60)
    
    X_test = test_df['text'].values
    y_test = test_df['label'].values
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    log_evaluation_results(logger, {
        'Accuracy': metrics['accuracy'],
        'F1 Macro': metrics['f1_macro'],
        'F1 Weighted': metrics['f1_weighted'],
        'Precision Macro': metrics['precision_macro'],
        'Recall Macro': metrics['recall_macro'],
        'MAE': metrics['mae'],
        "Cohen's Kappa": metrics['cohen_kappa'],
    }, title="Baseline Model - Test Results")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{metrics['confusion_matrix']}")
    
    logger.info("\nClassification Report:")
    logger.info(f"\n{metrics['classification_report']}")
    
    # Per-sample analysis
    results_df = analyze_per_sample(test_df, y_pred, y_proba, "Baseline Model", logger)
    summarize_difficult_samples(results_df, logger, top_n=20)
    summarize_confident_errors(results_df, logger, top_n=10)
    
    return metrics, results_df


def evaluate_transformer(model, tokenizer, test_df: pd.DataFrame, 
                         config: Dict, device: str, logger) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate transformer model on test data."""
    logger.info("=" * 60)
    logger.info("EVALUATING TRANSFORMER MODEL")
    logger.info("=" * 60)
    
    # Create dataset
    test_dataset = LegalTextDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        tokenizer,
        config['max_length']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Predict
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert back to 1-5 scale
    y_pred = np.array(all_preds) + 1
    y_test = np.array(all_labels) + 1
    y_proba = np.array(all_probs)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    log_evaluation_results(logger, {
        'Accuracy': metrics['accuracy'],
        'F1 Macro': metrics['f1_macro'],
        'F1 Weighted': metrics['f1_weighted'],
        'Precision Macro': metrics['precision_macro'],
        'Recall Macro': metrics['recall_macro'],
        'MAE': metrics['mae'],
        "Cohen's Kappa": metrics['cohen_kappa'],
    }, title="Transformer Model - Test Results")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{metrics['confusion_matrix']}")
    
    logger.info("\nClassification Report:")
    logger.info(f"\n{metrics['classification_report']}")
    
    # Per-sample analysis
    results_df = analyze_per_sample(test_df, y_pred, y_proba, "Transformer Model", logger)
    summarize_difficult_samples(results_df, logger, top_n=20)
    summarize_confident_errors(results_df, logger, top_n=10)
    
    return metrics, results_df



def analyze_per_sample(test_df: pd.DataFrame, y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray],
                       model_name: str, logger) -> pd.DataFrame:
    """Analyze model performance for each individual sample."""
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
        results_df['predicted_proba_true_class'] = [y_proba[i, label-1] for i, label in enumerate(results_df['label'])]
    
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
    logger.info(f"\nMost Common Confusions (True → Predicted):")
    confusions = results_df[~results_df['correct']].groupby(['label', 'predicted']).size().sort_values(ascending=False).head(10)
    for (true_label, pred_label), count in confusions.items():
        logger.info(f"  {true_label} → {pred_label}: {count} times")
    
    # ========================================================================
    # JAVÍTOTT: PER-LABEL PRECISION/RECALL/F1
    # ========================================================================
    from sklearn.metrics import precision_recall_fscore_support
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PER-LABEL METRICS: {model_name}")
    logger.info(f"{'='*60}")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        results_df['label'], 
        results_df['predicted'], 
        labels=[1, 2, 3, 4, 5],
        zero_division=0
    )
    
    # Formázott táblázat fejléc
    logger.info("")
    logger.info(f"{'Label':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    logger.info("-" * 60)
    
    # Sorok
    for i, label in enumerate([1, 2, 3, 4, 5]):
        logger.info(
            f"{label:<8} "
            f"{precision[i]:<12.3f} "
            f"{recall[i]:<12.3f} "
            f"{f1[i]:<12.3f} "
            f"{support[i]:<10}"
        )
    
    # Átlagok
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
    
    # ========================================================================
    # JAVÍTOTT: LABEL-ENKÉNTI HIBA ANALÍZIS
    # ========================================================================
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
    
    # ========================================================================
    # JAVÍTOTT: TOP 10 HARDEST SAMPLES (Kompakt)
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"TOP 10 HARDEST SAMPLES (Largest Errors)")
    logger.info(f"{'='*60}")
    
    hardest = results_df.nlargest(10, 'error').head(10)  # Csak 10!
    for idx, row in hardest.iterrows():
        text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
        conf_str = f", Conf: {row['confidence']:.3f}" if y_proba is not None else ""
        logger.info(
            f"{idx}. True={row['label']}, Pred={row['predicted']}, "
            f"Error={row['error']}{conf_str} | {text_preview}"
        )
    
    # ========================================================================
    # JAVÍTOTT: TOP 5 CONFIDENT MISTAKES (Kompakt)
    # ========================================================================
    if y_proba is not None:
        logger.info(f"\n{'='*60}")
        logger.info(f"TOP 5 CONFIDENT MISTAKES (High Confidence, Wrong)")
        logger.info(f"{'='*60}")
        
        mistakes = results_df[~results_df['correct']].nlargest(5, 'confidence')
        for idx, row in mistakes.iterrows():
            text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
            logger.info(
                f"{idx}. True={row['label']}, Pred={row['predicted']}, "
                f"Conf={row['confidence']:.3f} | {text_preview}"
            )
    
    return results_df



# def analyze_per_sample(test_df: pd.DataFrame, y_pred: np.ndarray,
#                        y_proba: Optional[np.ndarray],
#                        model_name: str, logger) -> pd.DataFrame:
#     """Analyze model performance for each individual sample."""
#     logger.info(f"\n{'='*60}")
#     logger.info(f"PER-SAMPLE ANALYSIS: {model_name}")
#     logger.info(f"{'='*60}")
    
#     # Create detailed per-sample dataframe
#     results_df = test_df.copy()
#     results_df['predicted'] = y_pred
#     results_df['error'] = np.abs(results_df['label'] - results_df['predicted'])
#     results_df['correct'] = results_df['label'] == results_df['predicted']
    
#     # Add prediction confidence if available
#     if y_proba is not None:
#         results_df['confidence'] = np.max(y_proba, axis=1)
#         results_df['predicted_proba_true_class'] = [y_proba[i, label-1] for i, label in enumerate(results_df['label'])]
    
#     # Overall statistics
#     logger.info(f"\nTotal samples: {len(results_df)}")
#     logger.info(f"Correct predictions: {results_df['correct'].sum()} ({results_df['correct'].mean()*100:.1f}%)")
#     logger.info(f"Average error: {results_df['error'].mean():.3f}")
#     logger.info(f"Max error: {results_df['error'].max():.0f}")
    
#     # Error distribution
#     logger.info(f"\nError Distribution:")
#     for error_val in sorted(results_df['error'].unique()):
#         count = (results_df['error'] == error_val).sum()
#         pct = count / len(results_df) * 100
#         logger.info(f"  Error {int(error_val)}: {count} samples ({pct:.1f}%)")
    
#     # Analyze by true label
#     logger.info(f"\nAccuracy by True Label:")
#     for label in sorted(results_df['label'].unique()):
#         subset = results_df[results_df['label'] == label]
#         accuracy = subset['correct'].mean()
#         logger.info(f"  Label {label}: {accuracy*100:.1f}% ({subset['correct'].sum()}/{len(subset)})")
    
#     # Most common confusions
#     logger.info(f"\nMost Common Confusions (True → Predicted):")
#     confusions = results_df[~results_df['correct']].groupby(['label', 'predicted']).size().sort_values(ascending=False).head(10)
#     for (true_label, pred_label), count in confusions.items():
#         logger.info(f"  {true_label} → {pred_label}: {count} times")
    
#     return results_df


def summarize_difficult_samples(results_df: pd.DataFrame, logger, top_n: int = 20) -> None:
    """Summarize the most difficult samples (highest errors)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TOP {top_n} MOST DIFFICULT SAMPLES")
    logger.info(f"{'='*60}")
    
    # Sort by error, then by confidence (if available)
    if 'confidence' in results_df.columns:
        difficult = results_df.nlargest(top_n, 'error').sort_values(['error', 'confidence'], ascending=[False, True])
    else:
        difficult = results_df.nlargest(top_n, 'error')
    
    for idx, (_, row) in enumerate(difficult.iterrows(), 1):
        logger.info(f"\n{idx}. True={row['label']}, Predicted={row['predicted']}, Error={row['error']:.0f}")
        if 'confidence' in results_df.columns:
            logger.info(f"   Confidence: {row['confidence']:.3f}")
        text_preview = row['text'][:150] + "..." if len(row['text']) > 150 else row['text']
        logger.info(f"   Text: {text_preview}")


def summarize_confident_errors(results_df: pd.DataFrame, logger, top_n: int = 10) -> None:
    """Summarize errors where the model was very confident but wrong."""
    if 'confidence' not in results_df.columns:
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TOP {top_n} CONFIDENT ERRORS (High Confidence, Wrong Prediction)")
    logger.info(f"{'='*60}")
    
    wrong_predictions = results_df[~results_df['correct']]
    if len(wrong_predictions) == 0:
        logger.info("No wrong predictions found.")
        return
    
    confident_errors = wrong_predictions.nlargest(top_n, 'confidence')
    
    for idx, (_, row) in enumerate(confident_errors.iterrows(), 1):
        logger.info(f"\n{idx}. True={row['label']}, Predicted={row['predicted']}, Confidence={row['confidence']:.3f}")
        text_preview = row['text'][:150] + "..." if len(row['text']) > 150 else row['text']
        logger.info(f"   Text: {text_preview}")


def plot_confusion_matrices(baseline_metrics: Dict, transformer_metrics: Dict,
                            output_dir: Path, logger) -> None:
    """Plot confusion matrices for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Baseline confusion matrix
    if baseline_metrics and 'confusion_matrix' in baseline_metrics:
        sns.heatmap(
            baseline_metrics['confusion_matrix'],
            annot=True, fmt='d', cmap='Blues',
            xticklabels=range(1, 6), yticklabels=range(1, 6),
            ax=axes[0]
        )
        axes[0].set_title('Baseline Model (TF-IDF + LogReg)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
    
    # Transformer confusion matrix
    if transformer_metrics and 'confusion_matrix' in transformer_metrics:
        sns.heatmap(
            transformer_metrics['confusion_matrix'],
            annot=True, fmt='d', cmap='Blues',
            xticklabels=range(1, 6), yticklabels=range(1, 6),
            ax=axes[1]
        )
        axes[1].set_title('Transformer Model (HuBERT)')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
    
    plt.tight_layout()
    
    output_path = output_dir / "confusion_matrices.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved confusion matrices to {output_path}")
    plt.close()


def plot_comparison(baseline_metrics: Dict, transformer_metrics: Dict,
                    output_dir: Path, logger) -> None:
    """Plot model comparison chart."""
    metrics_to_compare = ['accuracy', 'f1_macro', 'f1_weighted', 'cohen_kappa']
    labels = ['Accuracy', 'F1 Macro', 'F1 Weighted', "Cohen's Kappa"]
    
    baseline_values = [baseline_metrics.get(m, 0) for m in metrics_to_compare]
    transformer_values = [transformer_metrics.get(m, 0) for m in metrics_to_compare]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='steelblue')
    bars2 = ax.bar(x + width/2, transformer_values, width, label='Transformer', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison on Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = output_dir / "model_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved model comparison to {output_path}")
    plt.close()


def save_results(baseline_metrics: Dict, transformer_metrics: Dict,
                 output_dir: Path, logger) -> None:
    """Save evaluation results to JSON."""
    results = {
        'baseline': {
            'accuracy': baseline_metrics['accuracy'] if baseline_metrics else None,
            'f1_macro': baseline_metrics['f1_macro'] if baseline_metrics else None,
            'f1_weighted': baseline_metrics['f1_weighted'] if baseline_metrics else None,
            'mae': baseline_metrics['mae'] if baseline_metrics else None,
            'cohen_kappa': baseline_metrics['cohen_kappa'] if baseline_metrics else None,
        },
        'transformer': {
            'accuracy': transformer_metrics['accuracy'] if transformer_metrics else None,
            'f1_macro': transformer_metrics['f1_macro'] if transformer_metrics else None,
            'f1_weighted': transformer_metrics['f1_weighted'] if transformer_metrics else None,
            'mae': transformer_metrics['mae'] if transformer_metrics else None,
            'cohen_kappa': transformer_metrics['cohen_kappa'] if transformer_metrics else None,
        }
    }
    
    output_path = output_dir / "evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation results to {output_path}")


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
    
    # Load test data
    logger.info("\nLoading test data...")
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    if not test_path.exists():
        logger.error(f"Test data not found at {test_path}")
        logger.error("Please run 01_data_preprocessing.py first")
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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"\nUsing device: {device}")
    
    # Evaluate baseline model
    baseline_model = load_baseline_model(MODEL_DIR, logger)
    baseline_metrics = None
    baseline_results_df = None
    if baseline_model:
        baseline_metrics, baseline_results_df = evaluate_baseline(baseline_model, test_df, logger)
    
    # Evaluate transformer model
    transformer_model, tokenizer = load_transformer_model(MODEL_DIR, device, logger)
    transformer_metrics = None
    transformer_results_df = None
    if transformer_model:
        transformer_metrics, transformer_results_df = evaluate_transformer(
            transformer_model, tokenizer, test_df,
            TRANSFORMER_CONFIG, device, logger
        )
    
    # Model comparison summary
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    if baseline_metrics and transformer_metrics:
        logger.info(f"\n{'Metric':<20} {'Baseline':>12} {'Transformer':>12} {'Diff':>10}")
        logger.info("-" * 56)
        
        for metric in ['accuracy', 'f1_macro', 'f1_weighted', 'mae', 'cohen_kappa']:
            b_val = baseline_metrics.get(metric, 0)
            t_val = transformer_metrics.get(metric, 0)
            diff = t_val - b_val
            logger.info(f"{metric:<20} {b_val:>12.4f} {t_val:>12.4f} {diff:>+10.4f}")
    
    # Generate plots
    logger.info("\nGenerating evaluation plots...")
    output_dir = MODEL_DIR / "evaluation"
    output_dir.mkdir(exist_ok=True)
    
    if baseline_metrics and transformer_metrics:
        plot_confusion_matrices(baseline_metrics, transformer_metrics, output_dir, logger)
        plot_comparison(baseline_metrics, transformer_metrics, output_dir, logger)
    
    # Save results
    save_results(baseline_metrics, transformer_metrics, output_dir, logger)
    
    # Save detailed per-sample results
    if baseline_results_df is not None:
        baseline_csv_path = output_dir / "baseline_per_sample_results.csv"
        baseline_results_df.to_csv(baseline_csv_path, index=False)
        logger.info(f"\nSaved baseline per-sample results to {baseline_csv_path}")
    
    if transformer_results_df is not None:
        transformer_csv_path = output_dir / "transformer_per_sample_results.csv"
        transformer_results_df.to_csv(transformer_csv_path, index=False)
        logger.info(f"Saved transformer per-sample results to {transformer_csv_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
