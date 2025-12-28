from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib
import shap
from flask import Flask, request, jsonify
from flask import render_template
import numpy as np
np.random.seed(42)
from flasgger import Swagger

# --------------------------------------------
# Resolve project root
# --------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

# --------------------------------------------
# Load model and feature schema
# --------------------------------------------
model = joblib.load(BASE_DIR / "models" / "xgboost_model.pkl")
FEATURE_COLUMNS = joblib.load(BASE_DIR / "models" / "feature_columns.pkl")
# Force deterministic inference
model.set_params(
    n_jobs=1,
    random_state=42
)

# SHAP explainer
explainer = shap.TreeExplainer(model)

# --------------------------------------------
# Flask app + Swagger
# --------------------------------------------
app = Flask(__name__, template_folder="templates")

swagger = Swagger(app)

# --------------------------------------------
# Feature reconstruction
# --------------------------------------------
def build_features(input_json):
    now = datetime(2025, 1, 1, 12, 0, 0)


    row = {
        "temperature": input_json["temperature"],
        "vibration": input_json["vibration"],
        "pressure": input_json["pressure"],

        "hour": now.hour,
        "day": now.day,
        "day_of_week": now.weekday(),
        "month": now.month,

        "temp_mean_5": input_json["temperature"],
        "vib_std_10": 0.0,
        "pressure_max_30": input_json["pressure"],
        "failure_future": 0,

        "temp_roll_mean_2": input_json["temperature"],
        "vib_roll_mean_2": input_json["vibration"],
        "press_roll_mean_2": input_json["pressure"],

        "temp_roll_std_2": np.random.uniform(0.5, 1.2),
        "vib_roll_std_2": np.random.uniform(0.05, 0.15),
        "press_roll_std_2": np.random.uniform(0.2, 0.6),
    

        "temp_ema_2": input_json["temperature"],
        "vib_ema_2": input_json["vibration"],
        "press_ema_2": input_json["pressure"],

        "temp_roll_mean_3": input_json["temperature"],
        "vib_roll_mean_3": input_json["vibration"],
        "press_roll_mean_3": input_json["pressure"],

        "temp_roll_std_3": np.random.uniform(0.7, 1.5),
        "vib_roll_std_3": np.random.uniform(0.08, 0.25),
        "press_roll_std_3": np.random.uniform(0.3, 0.9),

        "temp_ema_3": input_json["temperature"],
        "vib_ema_3": input_json["vibration"],
        "press_ema_3": input_json["pressure"],
    }

    df = pd.DataFrame([row])
    df = df.reindex(columns=FEATURE_COLUMNS)
    return df

# --------------------------------------------
# Prediction logic
# --------------------------------------------
def predict_failure_logic(input_json):
    df = build_features(input_json)

    prob = float(model.predict_proba(df)[0][1])
    shap_values = explainer.shap_values(df)

    explanation = {
        feature: float(value)
        for feature, value in zip(df.columns, shap_values[0])
    }

    return {
        "status": "success",
        "failure_probability": round(prob, 4),
        "risk_level": (
            "HIGH" if prob > 0.25 else
            "MEDIUM" if prob > 0.10 else
            "LOW"
        ),
        "message": (
            "High risk of machine failure detected"
            if prob > 0.6 else
            "Moderate risk – monitor closely"
            if prob > 0.3 else
            "Low risk – machine operating normally"
        ),
        "shap_explanation": explanation
    }

# --------------------------------------------
# API Endpoints
# --------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return {"status": "FactoryGuardAI API running"}

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict machine failure risk
    ---
    tags:
      - Predictive Maintenance
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: sensor_data
        required: true
        schema:
          type: object
          properties:
            temperature:
              type: number
              example: 78
            vibration:
              type: number
              example: 0.62
            pressure:
              type: number
              example: 6.8
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            failure_probability:
              type: number
              example: 0.0347
            risk_level:
              type: string
              example: LOW
            message:
              type: string
              example: Low risk – machine operating normally
    """
    try:
        input_json = request.get_json()
        result = predict_failure_logic(input_json)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------
# Run server
# --------------------------------------------
@app.route("/ui")
def ui():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False)
