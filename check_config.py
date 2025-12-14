import sys
sys.path.insert(0, 'src')
from config import TRANSFORMER_CONFIG, BASELINE_CONFIG

print("=" * 60)
print("KONFIGURACIO ELLENORZES")
print("=" * 60)

print("\nTRANSFORMER CONFIG:")
print(f"  Augmentation: {TRANSFORMER_CONFIG.get('use_augmentation')}")
print(f"  Focal Loss: {TRANSFORMER_CONFIG.get('use_focal_loss')}")
print(f"  Freeze Layers: {TRANSFORMER_CONFIG.get('freeze_layers')}")
print(f"  Epochs: {TRANSFORMER_CONFIG.get('num_epochs')}")
print(f"  Batch Size: {TRANSFORMER_CONFIG.get('batch_size')}")
print(f"  Learning Rate: {TRANSFORMER_CONFIG.get('learning_rate')}")
print(f"  Dropout: {TRANSFORMER_CONFIG.get('dropout')}")

print("\nBASELINE CONFIG:")
print(f"  Augmentation: {BASELINE_CONFIG.get('use_augmentation')}")
print(f"  Classifier: {BASELINE_CONFIG.get('classifier')}")
print(f"  N-grams: {BASELINE_CONFIG.get('tfidf_ngram_range')}")

print("\n" + "=" * 60)
print("MINDEN KONFIGURACIO OK! ✓")
print("=" * 60)
