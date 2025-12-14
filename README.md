# Legal Text Decoder

## Deep Learning (VITMMA19) Project Work

---

## Project Information

| Field | Value |
|-------|-------|
| **Selected Topic** | Legal Text Decoder |
| **Student Name** | Petrich Tamás Ákos |
| **Neptun Code** | FA0B9B |
| **Aiming for +1 Mark** | Yes |

---

## Solution Description

### Problem Statement

A projekt célja egy természetes nyelvfeldolgozási (NLP) modell létrehozása, amely képes megjósolni, hogy egy adott Általános Szerződési Feltételek (ÁSZF) szövegének egy bekezdése mennyire könnyen vagy nehezen érthető egy átlagos felhasználó számára. A modell egy 1-től 5-ig terjedő skálán adja meg az érthetőséget.

### Rating Scale

| Rating | Description |
|--------|-------------|
| 1 | Nagyon nehezen vagy nem értelmezhető |
| 2 | Nehezen értelmezhető |
| 3 | Valamennyire érthető, de erősen kell koncentrálni |
| 4 | Végigolvasva megértem |
| 5 | Könnyen, egyből érthető |

### Model Architecture

A projekt inkrementális modellezési megközelítést alkalmaz:

#### 1. Baseline Model (TF-IDF + Logistic Regression)
- **Feature Extraction**: TF-IDF vectorizer (max 1500 features, unigrams + bigrams + trigrams)
- **Classifier**: Multinomial Logistic Regression with class balancing
- **Purpose**: Egyszerű referencia modell, amit a fejlettebb modellek próbálnak megverni

#### 2. Advanced Models (TF-IDF + Ensemble Methods)
- **XGBoost**: Gradient boosting regularizációval
- **Random Forest**: Döntési fák ensemble-je
- **Gradient Boosting**: Sklearn gradient boosting implementáció

Minden modell ugyanazt a TF-IDF feature extraction-t használja a fair összehasonlíthatóság érdekében.

### Training Methodology

1. **Data Loading**: Label Studio JSON exports feldolgozása
2. **Preprocessing**: Szöveg tisztítás, tokenizáció
3. **K-Fold Cross-Validation**: 5-fold stratified CV minden modellhez
4. **Training/Validation Split**: 85/15 stratified split
5. **Model Selection**: CV F1 Macro score alapján

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score (Macro/Weighted)**: Class-balanced performance
- **Mean Absolute Error (MAE)**: Rating prediction error
- **Cohen's Kappa**: Inter-rater agreement proxy
- **Confusion Matrix**: Detailed error analysis
- **Per-Label Precision/Recall/F1**: Label-szintű metrikák

---

## Extra Credit Justification

A következő elemek miatt pályázom a +1 jegyre:

1. **Inkrementális modellezés**: Baseline és többféle advanced modell összehasonlítása
2. **K-Fold Cross-Validation**: Megbízható teljesítmény becslés kis adathalmazon
3. **Teljes annotátor kiértékelés**: Minden annotátor minden értékelése külön kiértékelődik (~2500 teszt minta)
4. **Részletes kiértékelés**: Többféle metrika, confusion matrix, per-sample analysis
5. **Tiszta, moduláris kód**: Jól strukturált, dokumentált Python kód
6. **Docker kontainerizáció**: Teljes reprodukálhatóság
7. **Ensemble prediction**: Több modell kombinálása az inference során

---

## Data Preparation

### Data Source
- **Training**: FA0B9B neptun kódos mappa annotációi (~100 minta)
- **Test**: Consensus mappa összes annotációja (~2500 minta) - minden annotátor minden értékelése külön sample

### Processing Steps

1. **Download**: A SharePoint linkről automatikusan letöltjük az adatokat
2. **Extract**: ZIP fájl kicsomagolása
3. **Parse**: Label Studio JSON export formátum feldolgozása
4. **Clean**: Szöveg tisztítás (whitespace, special characters)
5. **All Annotations**: Teszt adatoknál minden annotáció külön kiértékelődik

### Data Format

**Input (Label Studio JSON)**:
```json
{
  "data": {"text": "Jogi szöveg..."},
  "annotations": [{"result": [{"value": {"choices": ["3-Többé/kevésbé megértem"]}}]}]
}
```

**Output (CSV)**:
```csv
text,label
"Jogi szöveg...",3
```

---

## Gyors Futtatás

### Módszer 1: Quick Start Script (Windows)

```powershell
.\quick_start.ps1
```

Ez automatikusan:
- ✅ Ellenőrzi a Python verziót
- ✅ Létrehozza a virtuális környezetet
- ✅ Telepíti a függőségeket
- ✅ Futtatja a teljes pipeline-t

### Módszer 2: Bash Script (Linux/Mac)

```bash
chmod +x run.sh
./run.sh
```

### Módszer 3: Docker

```bash
# Build
docker build -t legal-text-decoder .

# Run - teljes pipeline
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/log:/app/log \
  legal-text-decoder

# Windows PowerShell:
docker run --rm `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/log:/app/log `
  legal-text-decoder
```

---

## File Structure

```
LegalTextDecoder/
├── src/
│   ├── config.py              # Configuration and hyperparameters
│   ├── utils.py               # Utility functions and logging
│   ├── a01_data_preprocessing.py  # Data loading and preparation
│   ├── a02_training.py        # Model training (baseline + advanced)
│   ├── a03_evaluation.py      # Model evaluation on test set
│   └── a04_inference.py       # Prediction on new texts
├── notebook/
│   ├── 01_data_exploration.ipynb  # EDA and visualization
│   └── 02_label_analysis.ipynb    # Label distribution analysis
├── data/
│   ├── raw/                   # Downloaded data
│   └── processed/             # Prepared train/test CSVs
├── models/
│   ├── baseline_model.pkl     # Trained baseline (LogReg)
│   ├── xgboost_model.pkl      # Trained XGBoost
│   ├── random_forest_model.pkl    # Trained RandomForest
│   ├── gradient_boosting_model.pkl # Trained GradientBoosting
│   └── evaluation/            # Evaluation results and plots
├── log/
│   └── run.log               # Training and evaluation logs
├── Dockerfile                # Docker configuration
├── run.sh                    # Bash runner script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```


---

## Usage Examples

### Python API

```python
from src.a04_inference import load_models, predict_single
from src.config import MODEL_DIR, LOG_DIR
from src.utils import setup_logger

# Setup
logger = setup_logger("Demo", LOG_DIR / "run.log")

# Load all models
models = load_models(MODEL_DIR, logger)

# Predict
text = "A Szolgáltató fenntartja a jogot..."
result = predict_single(text, models)

print(f"Rating: {result['ensemble']['prediction']}")
print(f"Description: {result['ensemble']['description']}")

# Individual model predictions
for model_name, pred in result['predictions'].items():
    print(f"{model_name}: {pred['prediction']}")
```

### Command Line

```bash
# Single text
python src/a04_inference.py --text "Jogi szöveg..."

# File prediction
python src/a04_inference.py --input texts.txt --output predictions.csv

# Interactive mode
python src/a04_inference.py --interactive

# Use specific model only
python src/a04_inference.py --model baseline --text "Jogi szöveg..."
```

---

## Requirements

- Python 3.10+
- scikit-learn
- xgboost
- pandas, numpy
- matplotlib, seaborn
- See `requirements.txt` for full list

---

## License

This project was created for educational purposes as part of the Deep Learning (VITMMA19) course at BME.
