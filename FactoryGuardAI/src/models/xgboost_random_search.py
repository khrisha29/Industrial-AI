# ============================================
# Week 2 – Hyperparameter Optimization
# XGBoost with RandomizedSearchCV
# ============================================

from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, recall_score
import numpy as np

print(">>> RandomizedSearchCV for XGBoost started")

# --------------------------------------------
# Load data
# --------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["failure"]).select_dtypes(include=["number"])
y = df["failure"]

# Time-based split
split_index = int(len(df) * 0.8)
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

# --------------------------------------------
# Handle imbalance
# --------------------------------------------
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

# --------------------------------------------
# Model
# --------------------------------------------
xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight
)

# --------------------------------------------
# Hyperparameter space
# --------------------------------------------
param_dist = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9]
}

# --------------------------------------------
# RandomizedSearchCV
# --------------------------------------------
recall_scorer = make_scorer(recall_score)

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=10,              # small but sufficient
    scoring=recall_scorer,  # PDF-aligned
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

# --------------------------------------------
# Results
# --------------------------------------------
print("\n>>> Best Parameters Found:")
print(search.best_params_)

print("\n>>> Best Recall Score:")
print(search.best_score_)

print(">>> RandomizedSearchCV finished")
