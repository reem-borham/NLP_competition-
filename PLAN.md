# Vionex #DEEPX Hackathon — ABSA Plan

## Goal
Build an Arabic **Aspect-Based Sentiment Analysis (ABSA)** system for real customer reviews, producing:
- **Sentiment** (positive/negative or multi-class if ratings exist)
- **Aspect labels** (at minimum): `service`, `logistics`, `location`, `cleaning`

## 12-hour Hackathon Execution Plan

1) **Data + EDA (1–2h)**
- Load dataset, inspect columns and label formats.
- Clean/normalize Arabic text (remove diacritics/tatweel, normalize alef/yaa/taa marbuta).
- Check class balance, missingness, duplicates, length distributions.

2) **Aspect Labeling Strategy (2–3h)**
- Start with **weak labeling** (keyword rules) to create pseudo aspect labels.
- Manually validate a small sample per aspect and iterate keyword lists.
- (If time) move to a lightweight supervised multi-label aspect classifier.

3) **Modeling (4–6h)**
- Sentiment baseline: TF-IDF + linear classifier (LogReg/LinearSVC).
- Aspect baseline: multi-label one-vs-rest TF-IDF + linear models, or joint model.
- Upgrade path: Arabic transformer fine-tuning (AraBERT / MARBERT) if compute/time allows.

4) **Evaluation + Error Analysis (1–2h)**
- Metrics: F1 (macro) for sentiment; micro/macro F1 for aspects; per-aspect breakdown.
- Slice analysis by length, dialect, and frequent keywords.
- Inspect confusion examples and update preprocessing/labels.

5) **Packaging + Demo (1h)**
- Provide a simple inference function/API: input review → sentiment + aspects.
- Save artifacts; write a short README describing approach and limitations.

## Repo Assets
- `absa_aspect_labeling.py`: keyword-based weak aspect labeler (bootstrapping).
- `notebooks/vionex_deepx_absa_eda.ipynb`: Kaggle-friendly EDA + semantic EDA notebook.

