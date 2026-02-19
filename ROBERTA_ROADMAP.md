# RoBERTa + 3-Model Ensemble Roadmap

This workflow is fully separate from the DistilBERT files.

## Files Added

- `notebooks/phase2_roberta_macro_f1.ipynb`
- `notebooks/phase4_ensemble_3models_macro_f1.ipynb`
- `notebooks/phase2_deberta_macro_f1.ipynb`

## Run Order

1. **Phase 1 (TF-IDF baseline)**
	- Run: `python scripts/train_tfidf_macro_f1.py --n-splits 5 --seed 42`
	- Produces: `outputs/oof_probs.csv`, `outputs/test_probs.csv`

2. **Phase 2 DistilBERT (existing file, unchanged)**
	- Run existing DistilBERT notebook
	- Produces: `outputs/oof_probs_distilbert.csv`, `outputs/test_probs_distilbert.csv`

3. **Phase 2 RoBERTa (new file)**
	- Run `notebooks/phase2_roberta_macro_f1.ipynb`
	- Produces: `outputs/oof_probs_roberta.csv`, `outputs/test_probs_roberta.csv`, `outputs/submission_roberta.csv`

4. **Phase 4 3-model ensemble (new file)**
	- Run `notebooks/phase4_ensemble_3models_macro_f1.ipynb`
	- Produces final: `outputs/submission_ensemble_3models.csv`

5. **Phase 2 DeBERTa (new file)**
	- Run `notebooks/phase2_deberta_macro_f1.ipynb`
	- Produces: `outputs/oof_probs_deberta.csv`, `outputs/test_probs_deberta.csv`, `outputs/submission_deberta.csv`

If DeBERTa beats your current score, you can add it as a 4th model in a later ensemble step.

## Colab Tip

At the top of each notebook, run:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/GODS5.0_Hackathon
```

so all relative paths (`data_set/...`, `outputs/...`) work consistently.
