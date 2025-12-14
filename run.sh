#!/bin/bash
# =============================================================================
# Legal Text Decoder - Full Pipeline Runner
# =============================================================================
# This script runs the complete ML pipeline:
#   1. Data preprocessing
#   2. Model training (Baseline + Advanced models)
#   3. Model evaluation
#   4. Inference demo
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "LEGAL TEXT DECODER - Machine Learning Pipeline"
echo "============================================================"
echo ""
echo "Models: Baseline (LogReg), XGBoost, RandomForest, GradientBoosting"
echo ""

# Step 1: Data Preprocessing
echo "[1/4] Running data preprocessing..."
echo "------------------------------------------------------------"
python src/a01_data_preprocessing.py
echo ""

# Step 2: Training
echo "[2/4] Training models..."
echo "------------------------------------------------------------"
python src/a02_training.py
echo ""

# Step 3: Evaluation
echo "[3/4] Evaluating models..."
echo "------------------------------------------------------------"
python src/a03_evaluation.py
echo ""

# Step 4: Inference Demo
echo "[4/4] Running inference demo..."
echo "------------------------------------------------------------"
python src/a04_inference.py --text "A Szolgáltató fenntartja a jogot, hogy a szolgáltatást bármikor módosítsa vagy megszüntesse."
echo ""

echo "============================================================"
echo "Pipeline completed successfully!"
echo "============================================================"
