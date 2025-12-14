# Legal Text Decoder

## Deep Learning (VITMMA19) Project Work

---

## Project Information

| Field | Value |
|-------|-------|
| **Selected Topic** | Legal Text Decoder |
| **Student Name** | Petrich Tamás |
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

A projekt két fő modellt tartalmaz:

#### 1. Baseline Model (TF-IDF + Logistic Regression)
- **Feature Extraction**: TF-IDF vectorizer (max 5000 features, unigrams + bigrams)
- **Classifier**: Multinomial Logistic Regression with class balancing
- **Purpose**: Gyors, interprethálható baseline eredmények

#### 2. Transformer Model (HuBERT)
- **Pre-trained Model**: SZTAKI-HLT/hubert-base-cc (Hungarian BERT)
- **Architecture**: BERT encoder + classification head
- **Max Sequence Length**: 256 tokens
- **Training**: Fine-tuned with AdamW optimizer, linear warmup scheduler
- **Regularization**: Dropout (0.1), Early stopping, Class weighting

### Training Methodology

1. **Data Loading**: Label Studio JSON exports feldolgozása
2. **Preprocessing**: Szöveg tisztítás, tokenizáció
3. **Training/Validation Split**: 80/20 stratified split
4. **Training**: Cross-entropy loss with class weights
5. **Model Selection**: Early stopping based on validation F1 score

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score (Macro/Weighted)**: Class-balanced performance
- **Mean Absolute Error (MAE)**: Rating prediction error
- **Cohen's Kappa**: Inter-rater agreement proxy
- **Confusion Matrix**: Detailed error analysis

---

## Extra Credit Justification

A következő elemek miatt pályázom a +1 jegyre:

1. **Átfogó megoldás**: Baseline és fejlett transformer modell összehasonlítása
2. **Magyar nyelvi modell**: HuBERT fine-tuning specifikusan magyar jogi szövegekre
3. **Consensus alapú tesztelés**: Több annotátor egyetértésének figyelembevétele
4. **Részletes kiértékelés**: Többféle metrika, confusion matrix, model comparison
5. **Tiszta, moduláris kód**: Jól strukturált, dokumentált Python kód
6. **Docker kontainerizáció**: Teljes reprodukálhatóság

---

## Data Preparation

### Data Source
- **Training**: FA0B9B neptun kódos mappa annotációi
- **Test**: Consensus mappa (több annotátor közös címkézése)

### Processing Steps

1. **Download**: A SharePoint linkről automatikusan letöltjük az adatokat
2. **Extract**: ZIP fájl kicsomagolása
3. **Parse**: Label Studio JSON export formátum feldolgozása
4. **Clean**: Szöveg tisztítás (whitespace, special characters)
5. **Consensus Calculation**: Teszt adatoknál majority voting

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

## 🚀 Gyors Futtatás

### Módszer 1: Quick Start Script (Windows)

```powershell
.\quick_start.ps1
```

Ez automatikusan:
- ✅ Ellenőrzi a Python verziót
- ✅ Létrehozza a virtuális környezetet
- ✅ Telepíti a függőségeket
- ✅ Futtatja a teljes pipeline-t

### Módszer 2: Manuális (Lokálisan)

```powershell
# 1. Virtuális környezet
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Függőségek
pip install -r requirements.txt

# 3. Python path
$env:PYTHONPATH = "$PWD\src"

# 4. Futtatás
python main.py
```

**Várható futásidő**: 30-60 perc (CPU), 10-20 perc (GPU)

### Módszer 3: Docker (Bárhol)

```bash
# Build
docker build -t legal-text-decoder .

# Run - teljes pipeline
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/log:/app/log \
  legal-text-decoder

# Windows PowerShell:
docker run --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/log:/app/log `
  legal-text-decoder

# GPU támogatással
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  legal-text-decoder
```

### Lépésenkénti Futtatás

```powershell
# Csak adat feldolgozás
python src\a01_data_preprocessing.py

# Csak tanítás
python src\a02_training.py

# Csak kiértékelés
python src\a03_evaluation.py

# Inference egyetlen szövegre
python src\a04_inference.py --text "Az ÁSZF módosításáról e-mailben értesítjük."

# Interaktív mód
python src\a04_inference.py --interactive
```

**📖 Részletes útmutató**: Lásd [HOGYAN_FUTTASSAM.md](HOGYAN_FUTTASSAM.md)

---

## File Structure

```
LegalTextDecoder/
├── src/
│   ├── config.py              # Configuration and hyperparameters
│   ├── utils.py               # Utility functions and logging
│   ├── 01_data_preprocessing.py   # Data loading and preparation
│   ├── 02_training.py         # Model training (baseline + transformer)
│   ├── 03_evaluation.py       # Model evaluation on test set
│   └── 04_inference.py        # Prediction on new texts
├── notebook/
│   ├── 01_data_exploration.ipynb  # EDA and visualization
│   └── 02_label_analysis.ipynb    # Label distribution analysis
├── data/
│   ├── raw/                   # Downloaded data
│   └── processed/             # Prepared train/test CSVs
├── models/
│   ├── baseline_model.pkl     # Trained baseline model
│   └── transformer/           # Trained transformer model
├── log/
│   └── run.log               # Training and evaluation logs
├── Dockerfile                # Docker configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Configuration

Key hyperparameters (in `src/config.py`):

### Baseline Model
| Parameter | Value |
|-----------|-------|
| TF-IDF Max Features | 5000 |
| N-gram Range | (1, 2) |
| Classifier | Logistic Regression |
| Class Weight | Balanced |

### Transformer Model
| Parameter | Value |
|-----------|-------|
| Model | SZTAKI-HLT/hubert-base-cc |
| Max Length | 256 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Epochs | 10 |
| Warmup Ratio | 0.1 |
| Dropout | 0.1 |
| Early Stopping | 3 epochs |

---

## Results

*Results will be populated after training*

### Validation Set

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | MAE |
|-------|----------|------------|---------------|-----|
| Baseline | - | - | - | - |
| Transformer | - | - | - | - |

### Test Set (Consensus)

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | MAE |
|-------|----------|------------|---------------|-----|
| Baseline | - | - | - | - |
| Transformer | - | - | - | - |

---

## Usage Examples

### Python API

```python
from src.04_inference import load_models, predict_single

# Load models
baseline, transformer, tokenizer = load_models(MODEL_DIR, 'cuda', logger)

# Predict
text = "A Szolgáltató fenntartja a jogot..."
result = predict_single(text, baseline, transformer, tokenizer, 'cuda')

print(f"Rating: {result['ensemble']['prediction']}")
print(f"Description: {result['ensemble']['description']}")
```

### Command Line

```bash
# Single text
python src/04_inference.py --text "Jogi szöveg..."

# File prediction
python src/04_inference.py --input texts.txt --output predictions.csv

# Interactive mode
python src/04_inference.py --interactive
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU support)
- See `requirements.txt` for full list

---

## License

This project was created for educational purposes as part of the Deep Learning (VITMMA19) course at BME.
