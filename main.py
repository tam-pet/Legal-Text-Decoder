"""
Main entry point for Legal Text Decoder project.
Runs the complete pipeline: preprocessing -> training -> evaluation -> inference demo.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("=" * 60)
    print("LEGAL TEXT DECODER - Deep Learning Project")
    print("=" * 60)
    print()
    
    # Step 1: Data Preprocessing
    print("[1/4] Running data preprocessing...")
    print("-" * 40)
    from src.a01_data_preprocessing import main as preprocess_main
    try:
        preprocess_main()
    except Exception as e:
        print(f"Preprocessing error: {e}")
        print("Please ensure data is available in the data/raw directory")
    print()
    
    # Step 2: Training
    print("[2/4] Training models...")
    print("-" * 40)
    from src.a02_training import main as train_main
    try:
        train_main()
    except Exception as e:
        print(f"Training error: {e}")
    print()
    
    # Step 3: Evaluation
    print("[3/4] Evaluating models...")
    print("-" * 40)
    from src.a03_evaluation import main as eval_main
    try:
        eval_main()
    except Exception as e:
        print(f"Evaluation error: {e}")
    print()
    
    # Step 4: Inference Demo
    print("[4/4] Running inference demo...")
    print("-" * 40)
    from src.a04_inference import load_models, predict_single
    from src.config import MODEL_DIR, LOG_DIR
    from src.utils import setup_logger
    import torch
    
    logger = setup_logger("InferenceDemo", LOG_DIR / "run.log")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        baseline, transformer, tokenizer = load_models(MODEL_DIR, device, logger)
        
        # Demo texts
        demo_texts = [
            "A szolgáltatás használatához internetkapcsolat szükséges.",
            "A Szolgáltató fenntartja a jogot, hogy a szolgáltatást bármikor módosítsa vagy megszüntesse, előzetes értesítés nélkül.",
            "A jelen ÁSZF 12.3. pontjában meghatározott, a Ptk. 6:78. § (2) bekezdése szerinti elállási jog gyakorlásának határideje a termék átvételétől számított 14 nap.",
        ]
        
        print("\nInference Demo Results:")
        print("=" * 60)
        
        for i, text in enumerate(demo_texts, 1):
            result = predict_single(text, baseline, transformer, tokenizer, device)
            
            if 'ensemble' in result:
                pred = result['ensemble']
            elif 'transformer' in result:
                pred = result['transformer']
            elif 'baseline' in result:
                pred = result['baseline']
            else:
                continue
            
            print(f"\nText {i}: {text[:80]}...")
            print(f"Rating: {pred['prediction']} - {pred['description']}")
    
    except Exception as e:
        print(f"Inference error: {e}")
    
    print()
    print("=" * 60)
    print("Pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
