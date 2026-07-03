"""
DecodeLabs — Data Science Project 2
Supervised Learning: Leak-Free Classification Pipeline
--------------------------------------------------------
Dataset: ds_csv.csv (e-commerce order data)
Target:  IsCancelled (1 = Cancelled order, 0 = any other status)

NOTE ON DATASET MISMATCH
-------------------------
The brief is written for the classic credit-card fraud dataset
(284,807 transactions, 0.17% fraud rate). ds_csv.csv has no such
label and no comparable imbalance — OrderStatus is split ~evenly
across 5 classes (Cancelled ~21%). IsCancelled is used here as the
closest available proxy target so every required technique (SMOTE,
imblearn Pipeline, GridSearchCV, Precision/Recall/ROC-AUC) can still
be demonstrated correctly, even though real class imbalance is mild.

Pipeline architecture (per the brief):
  1. Feature preparation (leak-free)
  2. Stratified train/test split BEFORE any resampling/scaling
  3. imblearn.pipeline.Pipeline: [Scaler] -> SMOTE -> Classifier
  4. GridSearchCV tunes preprocessing + model together, safely
     re-applying SMOTE inside every CV fold
  5. Final evaluation on the untouched test set using Precision,
     Recall, F1, and ROC-AUC — never Accuracy
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, roc_curve
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ==================================================================
# 1. LOAD + PREPARE FEATURES (leak-free)
# ==================================================================
df = pd.read_csv('ds_csv.csv')
df['Date'] = pd.to_datetime(df['Date'])

# --- Target ---
df['IsCancelled'] = (df['OrderStatus'] == 'Cancelled').astype(int)

print("Target distribution:")
print(df['IsCancelled'].value_counts())
print("Positive class rate: {:.2f}%".format(df['IsCancelled'].mean() * 100))

# --- Feature engineering (same as Project 1, minus anything derived
#     from OrderStatus, which would leak the target) ---
df['OrderMonth'] = df['Date'].dt.month
df['OrderDayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['OrderDayOfWeek'].isin([5, 6]).astype(int)
df['PricePerItem'] = df['TotalPrice'] / df['ItemsInCart']
df['CartFillRatio'] = df['Quantity'] / df['ItemsInCart']
cust_orders = df.groupby('CustomerID').size().rename('CustomerOrderCount')
df = df.merge(cust_orders, on='CustomerID')
df['HasCoupon'] = df['CouponCode'].notna().astype(int)
df['CouponCode'] = df['CouponCode'].fillna('NONE')

# --- Outlier capping (IQR, vectorized) on numeric predictors only ---
for col in ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice']:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# --- Drop columns that would leak the target or carry no signal ---
# OrderStatus itself IS the target source -> drop.
# ID/free-text columns carry no generalizable signal -> drop.
leak_and_id_cols = ['OrderStatus', 'OrderID', 'TrackingNumber',
                     'ShippingAddress', 'CustomerID', 'Date']
df_model = df.drop(columns=leak_and_id_cols)

categorical_cols = ['Product', 'PaymentMethod', 'CouponCode', 'ReferralSource']
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

X = df_model.drop(columns=['IsCancelled'])
y = df_model['IsCancelled']

print("\nFinal feature matrix shape:", X.shape)

# ==================================================================
# 2. STRATIFIED TRAIN/TEST SPLIT — BEFORE any scaling or SMOTE
# ==================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print("\nTrain positive rate: {:.2f}% | Test positive rate: {:.2f}%".format(
    y_train.mean() * 100, y_test.mean() * 100
))

# ==================================================================
# 3. LEAK-FREE PIPELINES (imblearn, not sklearn)
#    SMOTE lives inside the pipeline so it is refit on the training
#    fold only, every time — the test set / validation fold never
#    sees synthetic rows.
# ==================================================================

# --- Logistic Regression: needs scaling (distance/gradient based) ---
lr_pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

lr_param_grid = {
    'smote__k_neighbors': [3, 5],
    'classifier__C': [0.01, 0.1, 1.0, 10],
}

# --- Random Forest: scale-invariant (splits are ordinal partitions) ---
rf_pipeline = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    'smote__k_neighbors': [3, 5],
    'classifier__n_estimators': [200, 400],
    'classifier__max_depth': [10, 20, None],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==================================================================
# 4. GRIDSEARCHCV — tunes preprocessing + model together, scored on
#    ROC-AUC (never accuracy) with cross-validation
# ==================================================================
print("\nTuning Logistic Regression...")
lr_search = GridSearchCV(lr_pipeline, lr_param_grid, scoring='roc_auc',
                          cv=cv, n_jobs=-1)
lr_search.fit(X_train, y_train)
print("Best LR params:", lr_search.best_params_)
print("Best LR CV ROC-AUC: {:.4f}".format(lr_search.best_score_))

print("\nTuning Random Forest...")
rf_search = GridSearchCV(rf_pipeline, rf_param_grid, scoring='roc_auc',
                          cv=cv, n_jobs=-1)
rf_search.fit(X_train, y_train)
print("Best RF params:", rf_search.best_params_)
print("Best RF CV ROC-AUC: {:.4f}".format(rf_search.best_score_))

# ==================================================================
# 5. FINAL EVALUATION — untouched test set, Precision/Recall/F1/ROC-AUC
# ==================================================================
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))
    print("Precision: {:.3f}".format(precision_score(y_test, y_pred)))
    print("Recall:    {:.3f}".format(recall_score(y_test, y_pred)))
    print("F1:        {:.3f}".format(f1_score(y_test, y_pred)))
    print("ROC-AUC:   {:.3f}".format(roc_auc_score(y_test, y_proba)))
    return {
        'model': name,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }

results = []
results.append(evaluate("Logistic Regression (SMOTE + Scaled)", lr_search.best_estimator_, X_test, y_test))
results.append(evaluate("Random Forest (SMOTE)", rf_search.best_estimator_, X_test, y_test))

results_df = pd.DataFrame(results)
print("\n\nFinal comparison:\n", results_df)

best_model_row = results_df.loc[results_df['roc_auc'].idxmax()]
print(f"\nBest model by ROC-AUC: {best_model_row['model']} ({best_model_row['roc_auc']:.3f})")

results_df.to_csv('model_comparison.csv', index=False)
print("\nSaved: model_comparison.csv")
