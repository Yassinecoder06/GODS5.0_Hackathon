# Two-Stage + Retrieval Stacking Roadmap

Standalone non-transformer-swap approach for significant score gains.

## File Added

- `notebooks/phase6_two_stage_retrieval_stacking.ipynb`

## Core Idea

1. Stage-1 binary ESG gate (`is_esg`)
2. Stage-2 E/S/G classifier only
3. Retrieval priors from nearest neighbors in TF-IDF space
4. Blend + threshold tuning for Macro-F1
5. Derive `non_ESG` from gate prediction

## Colab Startup

Run at top:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/GODS5.0_Hackathon
```

Then run the notebook top-to-bottom.

## Outputs

- `outputs/submission_two_stage_retrieval.csv`
- `outputs/two_stage_retrieval_report.json`
