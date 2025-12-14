"""
Inference script for Legal Text Decoder project.
================================================

This script runs predictions on new legal text paragraphs using trained models.

Features:
- Single text prediction
- Batch prediction from file
- Interactive mode
- Multi-model ensemble prediction

Usage:
    python src/a04_inference.py --text "Your legal text here"
    python src/a04_inference.py --input file.txt --output predictions.csv
    python src/a04_inference.py --interactive
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_DIR, LOG_DIR, NUM_CLASSES, MODELS_TO_TRAIN
from utils import setup_logger, clean_text


# =============================================================================
# Rating Descriptions
# =============================================================================

RATING_DESCRIPTIONS = {
    1: "Nagyon nehezen vagy nem értelmezhető",
    2: "Nehezen értelmezhető",
    3: "Valamennyire érthető, de erősen kell koncentrálni",
    4: "Végigolvasva megértem",
    5: "Könnyen, egyből érthető"
}


# =============================================================================
# Model Loading
# =============================================================================

def load_models(model_dir: Path, logger) -> Dict[str, Any]:
    """
    Load all trained models from disk.
    
    Args:
        model_dir: Directory containing model files
        logger: Logger instance
        
    Returns:
        Dictionary of model_name -> model
    """
    models = {}
    
    for model_name in MODELS_TO_TRAIN:
        model_path = model_dir / f"{model_name}_model.pkl"
        
        if model_path.exists():
            models[model_name] = joblib.load(model_path)
            logger.info(f"Loaded {model_name} model")
        else:
            logger.warning(f"Model not found: {model_path}")
    
    if not models:
        logger.warning("No models found!")
    
    return models


# =============================================================================
# Prediction Functions
# =============================================================================

def predict_single(
    text: str,
    models: Dict[str, Any],
    use_ensemble: bool = True
) -> Dict[str, Any]:
    """
    Make prediction for a single text using all available models.
    
    Args:
        text: Input text
        models: Dictionary of model_name -> model
        use_ensemble: Whether to compute ensemble prediction
        
    Returns:
        Dictionary with predictions from all models
    """
    text = clean_text(text)
    results = {'text': text, 'predictions': {}}
    
    all_proba = []
    
    for model_name, model in models.items():
        # Check if model uses XGBoost labels (0-4)
        use_xgb_labels = getattr(model, 'use_xgb_labels', False)
        display_name = getattr(model, 'display_name', model_name)
        
        # Make prediction
        pred = model.predict([text])[0]
        
        # Convert from 0-4 to 1-5 if XGBoost
        if use_xgb_labels:
            pred = pred + 1
        
        # Get probabilities if available
        proba = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba([text])[0]
                all_proba.append(proba)
            except Exception:
                pass
        
        results['predictions'][model_name] = {
            'prediction': int(pred),
            'description': RATING_DESCRIPTIONS[int(pred)],
            'display_name': display_name
        }
        
        if proba is not None:
            results['predictions'][model_name]['probabilities'] = {
                i+1: float(p) for i, p in enumerate(proba)
            }
    
    # Ensemble prediction (average probabilities)
    if use_ensemble and len(all_proba) > 1:
        avg_proba = np.mean(all_proba, axis=0)
        ensemble_pred = np.argmax(avg_proba) + 1
        
        results['ensemble'] = {
            'prediction': int(ensemble_pred),
            'description': RATING_DESCRIPTIONS[int(ensemble_pred)],
            'probabilities': {i+1: float(p) for i, p in enumerate(avg_proba)},
            'models_used': list(models.keys())
        }
    elif len(results['predictions']) == 1:
        # If only one model, use it as the "ensemble"
        single_model = list(results['predictions'].values())[0]
        results['ensemble'] = single_model.copy()
        results['ensemble']['models_used'] = list(models.keys())
    
    return results


def predict_batch(
    texts: List[str],
    models: Dict[str, Any],
    use_ensemble: bool = True
) -> List[Dict[str, Any]]:
    """
    Make predictions for a batch of texts.
    
    Args:
        texts: List of input texts
        models: Dictionary of model_name -> model
        use_ensemble: Whether to compute ensemble prediction
        
    Returns:
        List of prediction dictionaries
    """
    results = []
    
    for text in texts:
        result = predict_single(text, models, use_ensemble)
        results.append(result)
    
    return results


def predict_from_file(
    file_path: Path,
    models: Dict[str, Any],
    logger
) -> pd.DataFrame:
    """
    Make predictions for texts from a file.
    
    Supports:
    - .txt file with one paragraph per line
    - .csv file with 'text' column
    - .json file in Label Studio format
    
    Args:
        file_path: Path to input file
        models: Dictionary of model_name -> model
        logger: Logger instance
        
    Returns:
        DataFrame with predictions
    """
    logger.info(f"Processing file: {file_path}")
    
    texts = []
    
    if file_path.suffix == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        texts = df['text'].tolist()
    
    elif file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'data' in item:
                    texts.append(item['data'].get('text', ''))
                elif isinstance(item, str):
                    texts.append(item)
    
    logger.info(f"Found {len(texts)} texts to process")
    
    # Make predictions
    results = predict_batch(texts, models)
    
    # Create DataFrame
    rows = []
    for result in results:
        row = {'text': result['text']}
        
        # Add predictions from each model
        for model_name, pred in result.get('predictions', {}).items():
            row[f'{model_name}_pred'] = pred['prediction']
        
        # Add ensemble prediction
        if 'ensemble' in result:
            row['ensemble_pred'] = result['ensemble']['prediction']
            row['description'] = result['ensemble']['description']
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# Interactive Mode
# =============================================================================

def interactive_mode(models: Dict[str, Any], logger) -> None:
    """
    Run interactive prediction mode.
    
    Args:
        models: Dictionary of model_name -> model
        logger: Logger instance
    """
    logger.info("Starting interactive mode. Type 'quit' to exit.")
    
    print("\n" + "=" * 60)
    print("LEGAL TEXT DECODER - Interactive Mode")
    print("=" * 60)
    print(f"Loaded models: {', '.join(models.keys())}")
    print("Enter a legal text paragraph to analyze its readability.")
    print("Type 'quit' to exit.\n")
    
    while True:
        text = input("\nEnter text: ").strip()
        
        if text.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        result = predict_single(text, models)
        
        print("\n" + "-" * 40)
        print("PREDICTION RESULTS:")
        print("-" * 40)
        
        # Print each model's prediction
        for model_name, pred in result.get('predictions', {}).items():
            display = pred.get('display_name', model_name)
            print(f"\n{display}: {pred['prediction']} - {pred['description']}")
        
        # Print ensemble if available
        if 'ensemble' in result:
            pred = result['ensemble']
            print(f"\n>>> ENSEMBLE: {pred['prediction']} - {pred['description']}")
            
            if 'probabilities' in pred:
                print("\nProbability distribution:")
                for rating, prob in pred['probabilities'].items():
                    bar = "█" * int(prob * 20)
                    print(f"  {rating}: {bar} {prob:.2%}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(description="Legal Text Decoder Inference")
    parser.add_argument('--input', '-i', type=str, help='Input file path')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--text', '-t', type=str, help='Single text to predict')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--model', '-m', type=str, default='all',
                        help='Model to use (baseline, xgboost, random_forest, all)')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(
        name="Inference",
        log_file=LOG_DIR / "run.log",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("LEGAL TEXT DECODER - INFERENCE")
    logger.info("=" * 60)
    
    # Load models
    models = load_models(MODEL_DIR, logger)
    
    if not models:
        logger.error("No trained models found. Please run a02_training.py first.")
        return
    
    # Filter to specific model if requested
    if args.model != 'all' and args.model in models:
        models = {args.model: models[args.model]}
        logger.info(f"Using only {args.model} model")
    
    # Run inference
    if args.text:
        # Single text prediction
        result = predict_single(args.text, models)
        
        logger.info("\nPrediction Results:")
        if 'ensemble' in result:
            pred = result['ensemble']
            logger.info(f"Rating: {pred['prediction']} - {pred['description']}")
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.input:
        # File prediction
        input_path = Path(args.input)
        results_df = predict_from_file(input_path, models, logger)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_predictions.csv"
        
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Saved predictions to {output_path}")
        
        # Print summary
        print("\nPrediction Summary:")
        if 'ensemble_pred' in results_df.columns:
            print(results_df['ensemble_pred'].value_counts().sort_index())
    
    elif args.interactive or (not args.input and not args.text):
        # Interactive mode
        interactive_mode(models, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
