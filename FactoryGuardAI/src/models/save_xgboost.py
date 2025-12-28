# Save trained XGBoost model for Week 3 (SHAP)

from pathlib import Path
import pandas as pd
import joblib
from xgboost import XGBClassifier

print(">>> Training XGBoost model for persistence")

# Resolve project root
BASE_DIR = Path(__file__).resolve().parents[2]

# Load data
df = pd.read_csv(BASE_DIR / "data" / "processed" / "train.csv")

X = df.drop(columns=["failure"]).select_dtypes(include=["number"])
y = df["failure"]

# Time-based split (same as Week 2)
split_index = int(len(df) * 0.8)
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

# Handle imbalance
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

# XGBoost model (use your best-known params)
model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Save model
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "xgboost_model.pkl"
joblib.dump(model, model_path)

print(f">>> Model saved at: {model_path}")

# --------------------------------------------
# Save feature column order (CRITICAL for inference)
# --------------------------------------------
FEATURE_COLUMNS = X.columns.tolist()

feature_path = models_dir / "feature_columns.pkl"
joblib.dump(FEATURE_COLUMNS, feature_path)

print(f">>> Feature columns saved at: {feature_path}")
