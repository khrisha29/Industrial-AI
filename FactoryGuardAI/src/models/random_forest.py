

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print(">>> Random Forest script started")

# --------------------------------------------
# Resolve project root
# --------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

# --------------------------------------------
# Load processed data
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
# Train Random Forest
# --------------------------------------------
print(">>> Training Random Forest model...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predict probabilities instead of hard labels
y_prob = rf_model.predict_proba(X_test)[:, 1]

print(">>> Failure probabilities (first 20):")
print(y_prob[:20])

print(">>> Max predicted failure probability:", y_prob.max())

# Apply custom threshold
threshold = 0.2   # LOWER than default 0.5
y_pred = (y_prob >= threshold).astype(int)



print("\n=== Random Forest Results ===")
print(classification_report(y_test, y_pred))

print(">>> Random Forest script finished")
