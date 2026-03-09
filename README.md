# Sephora Premium vs Budget Price Tier Analysis

## Overview

This project predicts whether a Sephora product is **Premium** (above median price) or **Budget** using machine learning. It implements the full data science workflow: exploratory data analysis, model training with hyperparameter tuning, SHAP explainability, and an interactive Streamlit dashboard.

## Dataset

- **Source:** Sephora France product listings
- **Target:** Binary price tier (Premium vs Budget, split at median price)
- **Features:** 20 engineered features including ingredient counts, brand metrics, category encoding, and 13 key ingredient flags

## Models

| Model | Tuning Method |
|-------|--------------|
| Logistic Regression (Lasso/L1) | Baseline |
| Logistic Regression (Ridge/L2) | Baseline |
| Decision Tree (CART) | GridSearchCV |
| Random Forest | GridSearchCV |
| LightGBM | GridSearchCV |
| XGBoost | GridSearchCV |
| Neural Network (MLP) | Keras with early stopping |

## Repository Structure
├── sephora_products.csv # Raw dataset
├── sephora_price_tier.py # Full pipeline: EDA, training, SHAP
├── app.py # Streamlit dashboard
├── model_artifacts.pkl # Saved models, metrics, SHAP results
├── mlp_model.keras # Saved Keras MLP model
├── requirements.txt # Python dependencies
├── README.md # This file
├── eda_.png # EDA visualizations
├── model_.png # Model comparison plots
├── shap_*.png # SHAP analysis plots
└── mlp_training_history.png # MLP training curves

