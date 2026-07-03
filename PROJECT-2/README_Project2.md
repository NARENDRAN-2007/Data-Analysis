# Project 2 — Supervised Learning (Fraud/Anomaly Detection Pipeline)

DecodeLabs Data Science Industrial Training Kit — Batch 2026

## Important: dataset mismatch

The brief is written for the classic credit-card fraud dataset
(284,807 transactions, 0.17% fraud rate — extreme imbalance).
`ds_csv.csv` is the same e-commerce orders dataset used in Project 1.
It has **no fraud label**, and `OrderStatus` is split almost evenly
across 5 classes (Cancelled/Returned/Pending/Shipped/Delivered, each
~20%). There is no comparable class imbalance here.

`IsCancelled` (1 = order status is "Cancelled", 0 = anything else,
~21% positive) is used as the closest available proxy target so every
required technique — SMOTE, leak-free imblearn pipelines, GridSearchCV,
Precision/Recall/ROC-AUC — could still be demonstrated correctly on
real data, even though genuine imbalance and genuine predictive signal
are both mild-to-absent in this dataset (see Results below).

## Files

- `ds_csv.csv` — input dataset
- `project2_pipeline.py` — full pipeline script, run this
- `model_comparison.csv` — output: final metrics per model

## How to run

```
pip install pandas numpy scikit-learn imbalanced-learn
python project2_pipeline.py
```

## Pipeline architecture

1. **Feature preparation, leak-free** — reused Project 1's engineered
   features. Dropped `OrderStatus` (it's the target source), and all
   ID/free-text columns (`OrderID`, `TrackingNumber`,
   `ShippingAddress`, `CustomerID`, `Date`). One-hot encoded
   remaining categoricals.
2. **Stratified train/test split first** — 80/20, `stratify=y`, done
   *before* any scaling or SMOTE, so the test set reflects the real
   class ratio and never touches synthetic data.
3. **Leak-free pipelines** — built with `imblearn.pipeline.Pipeline`,
   not `sklearn.pipeline.Pipeline`, since sklearn's version silently
   drops or breaks on resampling steps:
   - Logistic Regression: `StandardScaler → SMOTE → LogisticRegression`
     (LR is scale-sensitive, so scaling is inside the pipeline)
   - Random Forest: `SMOTE → RandomForestClassifier`
     (tree splits are scale-invariant, so no scaler needed)
4. **GridSearchCV** — tunes `smote__k_neighbors` alongside each
   model's hyperparameters (`classifier__C` for LR;
   `classifier__n_estimators`, `classifier__max_depth` for RF), scored
   on **ROC-AUC**, with 5-fold stratified CV. Because SMOTE lives
   inside the pipeline, it's refit on the training fold only for every
   parameter combination — zero leakage into validation folds.
5. **Final evaluation** — on the untouched test set, using Precision,
   Recall, F1, and ROC-AUC. Accuracy is intentionally not used to
   select or report the "best" model.

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (SMOTE + Scaled) | 0.163 | 0.280 | 0.206 | 0.439 |
| Random Forest (SMOTE) | 0.182 | 0.080 | 0.111 | 0.397 |

**ROC-AUC ≈ 0.40–0.53 for both models — essentially random
(0.5 = no better than a coin flip).** This isn't a bug in the
pipeline; it's consistent with what Project 1's EDA already showed —
`OrderStatus` doesn't correlate meaningfully with product, payment
method, price, or any other feature in this dataset. There's no
learnable relationship for a classifier to find, because the
underlying data appears synthetically generated without an embedded
fraud/cancellation pattern.

## What this demonstrates vs. what it can't

**Correctly demonstrated:**
- SMOTE applied only inside the training fold, never leaking into
  test/validation data
- Scaler placement rule respected (inside pipeline, model-dependent)
- GridSearchCV tuning resampling + model hyperparameters jointly
- Evaluation strictly via Precision/Recall/F1/ROC-AUC, not Accuracy

**Can't be demonstrated on this data:**
- The core skill the brief is really testing — pulling a genuine rare
  signal out of a 0.17%-imbalance dataset — needs data where that
  signal actually exists.

## Recommendation

To actually exercise this pipeline the way the brief intends, run
`project2_pipeline.py` against the real Kaggle "Credit Card Fraud
Detection" dataset (`creditcard.csv`, 284,807 rows, `Class` column as
target, 0.17% fraud rate) — the column names would need small
adjustments (drop `Time`, use `Amount` + `V1`–`V28` as features,
`Class` as target), but the pipeline logic (split → leak-free
SMOTE/scale → GridSearchCV → Precision/Recall/ROC-AUC) is unchanged.
Want me to adapt the script for that dataset if you have it or can
download it?
