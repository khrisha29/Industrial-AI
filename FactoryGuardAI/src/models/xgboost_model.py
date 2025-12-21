# ============================================
# Week 2 – Day 2
# XGBoost Model (Final)
# ============================================

from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

print(">>> XGBoost script started")

# --------------------------------------------
# Resolve project root
# --------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

# --------------------------------------------
# Load processed dataset
# --------------------------------------------
DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"
print(f">>> Loading data from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(">>> Data loaded, shape:", df.shape)

# --------------------------------------------
# Prepare features and target
# --------------------------------------------
X = df.drop(columns=["failure"]).select_dtypes(include=["number"])
y = df["failure"]

print(">>> Feature matrix shape:", X.shape)
print(">>> Target distribution:")
print(y.value_counts())

# --------------------------------------------
# Time-based train-test split
# --------------------------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print(">>> Train size:", X_train.shape)
print(">>> Test size:", X_test.shape)

# --------------------------------------------
# Handle class imbalance (CRITICAL)
# --------------------------------------------
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(">>> scale_pos_weight:", scale_pos_weight)

# --------------------------------------------
# Train XGBoost model
# --------------------------------------------
print(">>> Training XGBoost model...")

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)

# --------------------------------------------
# Predictions
# --------------------------------------------
y_prob = xgb_model.predict_proba(X_test)[:, 1]

print(">>> Max predicted failure probability:", y_prob.max())
print(">>> Mean predicted failure probability:", y_prob.mean())
# --------------------------------------------
# Risk Scoring (IMPORTANT IMPROVEMENT)
# --------------------------------------------
risk_df = X_test.copy()
risk_df["failure_risk"] = y_prob

# Sort by highest risk first
risk_df = risk_df.sort_values("failure_risk", ascending=False)

print("\n>>> Top 10 highest-risk samples (risk scoring):")
print(risk_df[["failure_risk"]].head(10))

# Custom threshold (IMPORTANT)
threshold = 0.25
y_pred = (y_prob >= threshold).astype(int)

# --------------------------------------------
# Evaluation
# --------------------------------------------
print("\n=== XGBoost Results (threshold = 0.25) ===")
print(classification_report(y_test, y_pred))

print(">>> XGBoost script finished")
