# FactoryGuard AI – Predictive Maintenance Engine

This project implements an Industrial AI system for predictive maintenance using IoT sensor data.
The objective is to predict equipment failure 24 hours in advance using time-series features.

## Week 1 – Data Engineering & EDA
- Data ingestion and cleaning
- Missing value interpolation
- Lag feature creation
- Rolling window statistics
- Leakage-safe preprocessing
- Exploratory Data Analysis (EDA)
- Correlation matrix analysis

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost (planned)
- SHAP (planned)

## Project Structure
src/
data/
notebooks/

## Week 2 – Modeling & Evaluation

- Established a Logistic Regression baseline and observed severe class imbalance effects.
- Evaluated ensemble methods including Random Forest, which collapsed to the majority class under rare failure conditions.
- Implemented XGBoost with imbalance-aware weighting and probability-based evaluation.
- Reframed failure prediction as a risk-scoring problem by ranking high-risk machine states instead of relying on hard thresholds.
- Highlighted practical limitations of supervised learning under extreme rarity and discussed anomaly detection as future work.
