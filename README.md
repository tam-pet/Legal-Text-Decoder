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
- **Feature Extraction**: TF-IDF vectorizer (max 500 features, unigrams + bigrams)
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

---

## Gyors Futtatás

### Módszer: Docker

Run the following command in the root directory of the repository to build the Docker image:

```bash
# Build
docker build -t legal-text-decoder .

# Run - teljes pipeline
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/log:/app/log \
  legal-text-decoder > log/run.log 2>&1

# Windows PowerShell:
docker run --rm `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/log:/app/log `
  legal-text-decoder > log/run.log 2>&1
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

## Requirements

- Python 3.10+
- scikit-learn
- xgboost
- pandas, numpy
- matplotlib, seaborn
- See `requirements.txt` for full list

