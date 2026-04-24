# NLP_competition-

## Arabic ABSA (quick baseline)

To generate a new aspect label (`service`, `logistics`, `location`, `cleaning`) from Arabic hotel reviews using keyword-based weak labeling:

```bash
python3 absa_aspect_labeling.py --csv your_reviews.csv --out labeled_reviews.csv
```

If your dataset comes from `kagglehub`, use:

```bash
pip install kagglehub[pandas-datasets] pandas
python3 absa_aspect_labeling.py \
  --kaggle-dataset abedkhooli/arabic-100k-reviews \
  --kaggle-file <FILE_INSIDE_DATASET.csv> \
  --out labeled_reviews.csv
```
competition websit
d1y8zswxjnvm73.cloudfront.net
