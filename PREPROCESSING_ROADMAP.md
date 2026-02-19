# Preprocessing Ablation Roadmap

Run this first to validate whether preprocessing improves macro-F1 before expensive transformer retraining.

## Files Added

- `src/text_preprocessing.py`
- `notebooks/phase0_preprocessing_ablation.ipynb`

## Colab Startup

At the top of notebook runtime, use:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/GODS5.0_Hackathon
```

## What the Notebook Does

1. Loads train/test data
2. Runs TF-IDF CV macro-F1 ablation over preprocessing configs (`raw`, `basic_clean`, `esg_normalized`)
3. Shows per-label and macro-F1 comparison
4. Exports best cleaned datasets for downstream training

## Output Files

- `outputs/preprocessing_ablation_report.csv`
- `outputs/preprocessing_examples.csv`
- `outputs/train_cleaned_esg_normalized.csv` (or best config name)
- `outputs/test_cleaned_esg_normalized.csv` (or best config name)

## Next Step Integration

After identifying best preprocessing config, apply that config in your RoBERTa/DistilBERT notebooks before tokenization and retrain.
