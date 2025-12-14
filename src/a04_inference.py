"""
Inference script for Legal Text Decoder project.
Runs prediction on new legal text paragraphs.
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_DIR, LOG_DIR, TRANSFORMER_CONFIG, NUM_CLASSES
from utils import setup_logger, clean_text
from a02_training import LegalTextDataset, TransformerClassifier


# Rating descriptions
RATING_DESCRIPTIONS = {
    1: "Nagyon nehezen vagy nem értelmezhető",
    2: "Nehezen értelmezhető",
    3: "Valamennyire érthető, de erősen kell koncentrálni",
    4: "Végigolvasva megértem",
    5: "Könnyen, egyből érthető"
}


def load_models(model_dir: Path, device: str, logger) -> Tuple[Any, Any, Any]:
    """Load trained models."""
    # Load baseline model
    baseline_path = model_dir / "baseline_model.pkl"
    baseline_model = None
    if baseline_path.exists():
        baseline_model = joblib.load(baseline_path)
        logger.info("Loaded baseline model")
    
    # Load transformer model
    transformer_dir = model_dir / "transformer"
    transformer_model = None
    tokenizer = None
    
    if transformer_dir.exists():
        config_path = transformer_dir / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
        
        transformer_model = TransformerClassifier(
            config['model_name'],
            NUM_CLASSES,
            config.get('dropout', 0.1)
        )
        transformer_model.load_state_dict(
            torch.load(transformer_dir / "model.pt", map_location=device)
        )
        transformer_model = transformer_model.to(device)
        transformer_model.eval()
        logger.info("Loaded transformer model")
    
    return baseline_model, transformer_model, tokenizer


def predict_single(text: str, baseline_model, transformer_model, 
                   tokenizer, device: str) -> Dict[str, Any]:
    """
    Make prediction for a single text.
    
    Returns:
        Dictionary with predictions from both models
    """
    text = clean_text(text)
    results = {'text': text}
    
    # Baseline prediction
    if baseline_model:
        baseline_pred = baseline_model.predict([text])[0]
        baseline_proba = baseline_model.predict_proba([text])[0]
        
        results['baseline'] = {
            'prediction': int(baseline_pred),
            'description': RATING_DESCRIPTIONS[baseline_pred],
            'probabilities': {i+1: float(p) for i, p in enumerate(baseline_proba)}
        }
    
    # Transformer prediction
    if transformer_model and tokenizer:
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=TRANSFORMER_CONFIG['max_length'],
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = transformer_model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs) + 1  # Convert 0-4 to 1-5
        
        results['transformer'] = {
            'prediction': int(pred),
            'description': RATING_DESCRIPTIONS[pred],
            'probabilities': {i+1: float(p) for i, p in enumerate(probs)}
        }
    
    # Ensemble prediction (average)
    if 'baseline' in results and 'transformer' in results:
        avg_probs = np.zeros(5)
        for i in range(5):
            avg_probs[i] = (results['baseline']['probabilities'][i+1] + 
                          results['transformer']['probabilities'][i+1]) / 2
        
        ensemble_pred = np.argmax(avg_probs) + 1
        results['ensemble'] = {
            'prediction': int(ensemble_pred),
            'description': RATING_DESCRIPTIONS[ensemble_pred],
            'probabilities': {i+1: float(p) for i, p in enumerate(avg_probs)}
        }
    
    return results


def predict_batch(texts: List[str], baseline_model, transformer_model,
                  tokenizer, device: str, batch_size: int = 32) -> List[Dict[str, Any]]:
    """Make predictions for a batch of texts."""
    results = []
    
    for text in texts:
        result = predict_single(text, baseline_model, transformer_model, tokenizer, device)
        results.append(result)
    
    return results


def predict_from_file(file_path: Path, baseline_model, transformer_model,
                      tokenizer, device: str, logger) -> pd.DataFrame:
    """
    Make predictions for texts from a file.
    
    Supports:
    - .txt file with one paragraph per line
    - .csv file with 'text' column
    - .json file in Label Studio format
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
    results = predict_batch(texts, baseline_model, transformer_model, tokenizer, device)
    
    # Create DataFrame
    rows = []
    for result in results:
        row = {'text': result['text']}
        
        if 'baseline' in result:
            row['baseline_pred'] = result['baseline']['prediction']
        
        if 'transformer' in result:
            row['transformer_pred'] = result['transformer']['prediction']
        
        if 'ensemble' in result:
            row['ensemble_pred'] = result['ensemble']['prediction']
            row['description'] = result['ensemble']['description']
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def interactive_mode(baseline_model, transformer_model, tokenizer, 
                     device: str, logger) -> None:
    """Run interactive prediction mode."""
    logger.info("Starting interactive mode. Type 'quit' to exit.")
    print("\n" + "=" * 60)
    print("LEGAL TEXT DECODER - Interactive Mode")
    print("=" * 60)
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
        
        result = predict_single(text, baseline_model, transformer_model, tokenizer, device)
        
        print("\n" + "-" * 40)
        print("PREDICTION RESULTS:")
        print("-" * 40)
        
        if 'baseline' in result:
            pred = result['baseline']
            print(f"\nBaseline Model: {pred['prediction']} - {pred['description']}")
        
        if 'transformer' in result:
            pred = result['transformer']
            print(f"Transformer Model: {pred['prediction']} - {pred['description']}")
        
        if 'ensemble' in result:
            pred = result['ensemble']
            print(f"\n>>> Ensemble: {pred['prediction']} - {pred['description']}")
            print("\nProbability distribution:")
            for rating, prob in pred['probabilities'].items():
                bar = "█" * int(prob * 20)
                print(f"  {rating}: {bar} {prob:.2%}")


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(description="Legal Text Decoder Inference")
    parser.add_argument('--input', '-i', type=str, help='Input file path')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--text', '-t', type=str, help='Single text to predict')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--model', '-m', type=str, default='ensemble',
                        choices=['baseline', 'transformer', 'ensemble'],
                        help='Model to use for prediction')
    
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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load models
    baseline_model, transformer_model, tokenizer = load_models(MODEL_DIR, device, logger)
    
    if not baseline_model and not transformer_model:
        logger.error("No trained models found. Please run 02_training.py first.")
        return
    
    # Run inference
    if args.text:
        # Single text prediction
        result = predict_single(args.text, baseline_model, transformer_model, tokenizer, device)
        
        logger.info("\nPrediction Results:")
        if 'ensemble' in result:
            pred = result['ensemble']
            logger.info(f"Rating: {pred['prediction']} - {pred['description']}")
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.input:
        # File prediction
        input_path = Path(args.input)
        results_df = predict_from_file(
            input_path, baseline_model, transformer_model, tokenizer, device, logger
        )
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_predictions.csv"
        
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Saved predictions to {output_path}")
        
        # Print summary
        print("\nPrediction Summary:")
        print(results_df.describe())
    
    elif args.interactive or (not args.input and not args.text):
        # Interactive mode
        interactive_mode(baseline_model, transformer_model, tokenizer, device, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
