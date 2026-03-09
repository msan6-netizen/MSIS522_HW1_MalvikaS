# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University project (MSIS522) that predicts whether a Sephora product is **Premium** or **Budget** (binary classification split at median price) using 7 ML models. The end product is a Streamlit dashboard deployed for interactive exploration.

## Architecture

Two main Python files form the entire pipeline:

- **`sephora_price_tier.py`** — Offline training pipeline. Loads `sephora_products.csv`, engineers 20 features (ingredient flags, brand metrics, category encoding), trains 7 models (Logistic Lasso/Ridge, CART, Random Forest, LightGBM, XGBoost, Keras MLP) with GridSearchCV tuning, generates EDA/SHAP plots, and serializes everything into `model_artifacts.pkl` + `mlp_model.keras`.
- **`app.py`** — Streamlit dashboard. Loads pre-built artifacts and renders 4 tabs: Executive Summary, Descriptive Analytics, Model Performance, and Explainability & Interactive Prediction (with live SHAP waterfall).

Data flows one-way: `sephora_price_tier.py` produces artifacts → `app.py` consumes them. The dashboard never retrains models.

## Hardcoded Paths

Both files use hardcoded Windows paths (`C:\Malvika\sephora_dashboard`). When deploying to Streamlit Cloud or another environment, these paths must be changed to relative paths (e.g., `./` or use `pathlib`). This is the most likely source of deployment failures.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
# Note: tensorflow/keras is also required for the training script but not in requirements.txt

# Run the training pipeline (generates all artifacts and plots)
python sephora_price_tier.py

# Run the Streamlit dashboard locally
streamlit run app.py

# Run via devcontainer (Codespaces)
# Automatically starts on port 8501 via postAttachCommand
```

## Key Technical Details

- Python 3.11 (specified in `.python-version`)
- Linear models (Logistic Lasso/Ridge) use `StandardScaler`; tree-based models use raw features. The `app.py` interactive predictor handles this split via the `is_linear` check.
- `model_artifacts.pkl` contains fitted models, SHAP explainers, scalers, label encoders, CV results, and dataset metadata — everything the dashboard needs.
- The MLP model is saved separately as `mlp_model.keras` but is **not** loaded in `app.py` (only the 6 sklearn-compatible models in `fitted_models` are used for interactive prediction).
- Pre-generated PNG plots are loaded as static images in the dashboard rather than rendered dynamically.
