# FactoryGuard AI – Predictive Maintenance Engine
FactoryGuard AI is an end-to-end Industrial AI system for predictive maintenance using IoT sensor data.
The system predicts equipment failure risk in advance using time-series feature engineering, machine learning, explainability, and deployment as a real-time service.

## Problem Statement
Unexpected machine failures cause costly downtime in industrial environments.
This project aims to predict failure risk early using sensor signals (temperature, vibration, pressure) and provide explainable predictions that engineers can trust.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Flask
- Joblib

## Week 1 – Data Engineering & EDA
- Ingested and cleaned raw IoT sensor data
- Handled missing values using interpolation
- Created lag features and rolling window statistics
- Ensured leakage-safe preprocessing for time-series data
- Performed exploratory data analysis and correlation analysis

## Week 2 – Modeling & Evaluation
- Built a Logistic Regression baseline to highlight class imbalance
- Evaluated Random Forest and observed majority-class collapse
- Implemented XGBoost with imbalance-aware weighting
- Framed failure prediction as a risk-scoring problem instead of binary classification
- Focused evaluation on recall and F1-score rather than accuracy

## Week 3 – Model Explainability
- Integrated SHAP to explain XGBoost predictions
- Generated local explanations for individual failure predictions
- Identified key sensor features contributing to high-risk scenarios
- Validated that model behavior aligns with real industrial failure patterns

## Week 4 – Model-as-a-Service Deployment
- Serialized the trained model and feature schema using joblib
- Built a Flask-based REST API for real-time inference
- Implemented a /predict endpoint accepting JSON sensor inputs
- Returned failure probability, risk level, and SHAP-based explanations
- Added a simple interactive UI for demonstration
- Ensured deterministic inference for stable and reproducible outputs

## To Run:
## Activate environment
.\.venv\Scripts\Activate
## Start the API
python src/inference/predict_failure.py
## Open in browser:
http://127.0.0.1:5000/ui

## Final Outcome
FactoryGuard AI delivers a complete industrial AI pipeline from raw sensor data and feature engineering to explainable, deployable, real-time failure risk prediction aligned with real-world predictive maintenance requirements.
