# ============================================
# Week 2 – Day 1
# Baseline Modeling: Logistic Regression
# Project: FactoryGuard AI (Predictive Maintenance)
# ============================================

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --------------------------------------------
# STEP 1: Resolve project root safely
# --------------------------------------------
# This ensures the script works no matter
# where it is run from (production-grade)
BASE_DIR = Path(__file__).resolve().parents[2]

# --------------------------------------------
# STEP 2: Load processed dataset (Week 1 output)
# --------------------------------------------
DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"

print(f"Loading data from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# --------------------------------------------
# STEP 3: Separate features and target
# --------------------------------------------
# IMPORTANT:
# - Logistic Regression needs numeric data
# - Timestamp & strings must be removed
# - Temporal info is already captured via
#   lag & rolling features (Week 1)
X = df.drop(columns=["failure"]).select_dtypes(include=["number"])
y = df["failure"]

print("Numeric feature columns used for modeling:")
print(X.columns.tolist())
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# --------------------------------------------
# STEP 4: Time-based train-test split
# --------------------------------------------
# NO random split (prevents data leakage)
split_ratio = 0.8
split_index = int(len(df) * split_ratio)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# --------------------------------------------
# STEP 5: Baseline Logistic Regression
# --------------------------------------------
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)

print("\n=== Baseline Logistic Regression Results ===")
print(classification_report(y_test, y_pred))

# --------------------------------------------
# STEP 6: Logistic Regression with class balancing
# --------------------------------------------
# Handles severe class imbalance (failures are rare)
balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

balanced_model.fit(X_train, y_train)

y_pred_balanced = balanced_model.predict(X_test)

print("\n=== Balanced Logistic Regression Results ===")
print(classification_report(y_test, y_pred_balanced))

# --------------------------------------------
# STEP 7: Observations (for evaluators)
# --------------------------------------------
"""
Observations:
- Baseline Logistic Regression is biased toward the majority (non-failure) class.
- Due to severe class imbalance, accuracy alone is misleading.
- Introducing class_weight='balanced' improves recall for failure events.
- In predictive maintenance, missing a failure is costlier than a false alarm,
  hence recall and F1-score are more meaningful metrics.
"""
