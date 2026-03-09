"""
Sephora Premium vs Budget — Full Pipeline
==========================================
Trains 7 models (Lasso, Ridge, CART, RF, LGBM, XGB, MLP) with
cross-validation / GridSearchCV, generates EDA plots and SHAP analysis.
"""

import pandas as pd
import numpy as np
import json
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, cross_validate,
                                     StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
OUTDIR = str(Path(__file__).resolve().parent)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PARSE
# ═══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(f'{OUTDIR}/sephora_products.csv')
print(f"Loaded rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Sample row:\n{df.iloc[0].to_dict()}")


# -- Price: convert to float (handle $ signs, commas, etc.)
df['price'] = (df['price'].astype(str)
               .str.replace(r'[^\d.]', '', regex=True)
               .replace('', np.nan)
               .astype(float))
df = df.dropna(subset=['price'])

# -- Brand: already a plain string column
df['brand_name'] = df['brand'].fillna('Unknown')

# -- Reviews & likes (use as proxy features since no rating column)
df['reviews_count'] = pd.to_numeric(df['reviews_count'], errors='coerce').fillna(0)
df['likes_count'] = pd.to_numeric(df['likes_count'], errors='coerce').fillna(0)

print("Sample breadcrumbs:")
print(df['breadcrumbs'].head(5).tolist())


# -- Category from breadcrumbs (e.g. "Makeup > Lips > Lipstick")
def extract_category_from_url(url):
    """Extract a rough category from the product URL or name."""
    if not isinstance(url, str):
        return 'Other'
    url_lower = url.lower()
    if any(w in url_lower for w in ['lipstick', 'lip-', 'lips', 'gloss']):
        return 'Lips'
    elif any(w in url_lower for w in ['eye', 'liner', 'mascara', 'shadow', 'brow']):
        return 'Eyes'
    elif any(w in url_lower for w in ['foundation', 'concealer', 'powder', 'primer', 'blush', 'bronzer', 'contour', 'highlight']):
        return 'Face'
    elif any(w in url_lower for w in ['cleanser', 'moistur', 'serum', 'cream', 'mask', 'toner', 'sunscreen', 'spf', 'skincare', 'acne', 'retinol', 'peel']):
        return 'Skincare'
    elif any(w in url_lower for w in ['perfume', 'cologne', 'fragrance', 'eau-de']):
        return 'Fragrance'
    elif any(w in url_lower for w in ['shampoo', 'conditioner', 'hair']):
        return 'Hair'
    elif any(w in url_lower for w in ['nail', 'polish']):
        return 'Nails'
    elif any(w in url_lower for w in ['brush', 'sponge', 'tool', 'applicator']):
        return 'Tools'
    else:
        return 'Other'

df['category'] = df['url'].apply(extract_category_from_url)



# -- Ingredients: use 'ingrediat_desc' or 'raw_ingrediat_desc'
df['ingredients'] = df['ingrediat_desc'].fillna(df.get('raw_ingrediat_desc', pd.Series(dtype=str))).fillna('')

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

median_price = df['price'].median()
df['price_tier'] = (df['price'] > median_price).astype(int)

key_ingredients = {
    'retinol': r'retinol|retinal|retinoic',
    'vitamin_c': r'ascorb|vitamin c',
    'hyaluronic_acid': r'hyaluron',
    'niacinamide': r'niacinamide',
    'salicylic_acid': r'salicylic',
    'glycolic_acid': r'glycolic',
    'peptides': r'peptide',
    'ceramides': r'ceramide',
    'squalane': r'squalane|squalene',
    'collagen': r'collagen',
    'aha_bha': r'\baha\b|\bbha\b',
    'spf': r'\bspf\b|sunscreen|sun filter',
    'fragrance': r'parfum|fragrance',
}

for feat_name, pattern in key_ingredients.items():
    df[f'has_{feat_name}'] = df['ingredients'].str.lower().str.contains(pattern, regex=True).astype(int)

df['ingredient_count'] = df['ingredients'].apply(
    lambda x: len([i.strip() for i in re.split(r'[,;]', x) if i.strip()]) if x else 0
)

brand_counts = df['brand_name'].value_counts()
df['brand_frequency'] = df['brand_name'].map(brand_counts).fillna(0)

luxury_brands = ['DIOR', 'TOM FORD', 'GUERLAIN', 'YVES SAINT LAURENT', 'GIVENCHY',
                 'CHANEL', 'LANCÔME', 'MAISON MARGIELA', 'GIORGIO ARMANI',
                 'VALENTINO', 'BURBERRY', 'HERMÈS', 'PRADA']
df['is_luxury_brand'] = df['brand_name'].isin(luxury_brands).astype(int)

df['out_of_stock_flag'] = df['out_of_stock'].apply(
    lambda x: 1 if str(x).strip().lower() in ('true', '1', 'yes') else 0
)
df['log_reviews'] = np.log1p(df['reviews_count'])
df['log_likes'] = np.log1p(df['likes_count'])

# Keep top categories (at least 20 products), group rest as 'Other'
main_categories = df['category'].unique().tolist()

print(f"After category filter: {len(df)} rows")
print(f"Price tier distribution:\n{df['price_tier'].value_counts()}")

le_cat = LabelEncoder()
df['category_encoded'] = le_cat.fit_transform(df['category'])

print(f"Dataset: {len(df)} products | Median price: ${median_price:.2f}")
print(f"Target: {df['price_tier'].value_counts().rename({0:'Budget',1:'Premium'}).to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. EDA VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

sns.set_theme(style="whitegrid", palette="muted")

# 3a. Price distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['price'], bins=40, color='#4A90D9', edgecolor='white', alpha=0.85)
axes[0].axvline(median_price, color='red', linestyle='--', linewidth=2, label=f'Median: ${median_price:.0f}')
axes[0].set_xlabel('Price ($)')
axes[0].set_ylabel('Count')
axes[0].set_title('Price Distribution')
axes[0].legend()

tier_counts = df['price_tier'].value_counts().rename({0:'Budget', 1:'Premium'})
axes[1].bar(tier_counts.index, tier_counts.values, color=['#5DADE2', '#E74C3C'], edgecolor='white')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(tier_counts.index.tolist())
axes[1].set_ylabel('Count')
axes[1].set_title('Target Variable: Price Tier')
for i, v in enumerate(tier_counts.values):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/eda_price_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_price_dist.png")

# 3b. Price by category (boxplot)
fig, ax = plt.subplots(figsize=(10, 6))
order = df.groupby('category')['price'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='category', y='price', order=order, palette='coolwarm', ax=ax)
ax.set_title('Price Distribution by Category')
ax.set_xlabel('')
ax.set_ylabel('Price ($)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/eda_price_by_category.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_price_by_category.png")

# 3c. Luxury vs non-luxury
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='is_luxury_brand', y='price', palette=['#5DADE2', '#E74C3C'], ax=ax)
ax.set_xticklabels(['Non-Luxury', 'Luxury'])
ax.set_title('Price: Luxury vs Non-Luxury Brands')
ax.set_ylabel('Price ($)')
ax.set_xlabel('')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/eda_luxury_vs_nonluxury.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_luxury_vs_nonluxury.png")

# 3d. Ingredient count vs price tier
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(data=df, x='price_tier', y='ingredient_count', palette=['#5DADE2', '#E74C3C'], ax=ax)
ax.set_xticklabels(['Budget', 'Premium'])
ax.set_title('Ingredient Complexity by Price Tier')
ax.set_ylabel('Number of Ingredients')
ax.set_xlabel('')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/eda_ingredients_by_tier.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_ingredients_by_tier.png")

# 3e. Correlation heatmap of features
feature_cols = [
    'category_encoded', 'ingredient_count', 'brand_frequency', 'is_luxury_brand',
    'out_of_stock_flag', 'log_reviews', 'log_likes',
] + [f'has_{k}' for k in key_ingredients.keys()]

feature_display_names = {
    'category_encoded': 'Category',
    'ingredient_count': 'Ingr. Count',
    'brand_frequency': 'Brand Pop.',
    'is_luxury_brand': 'Luxury',
    'out_of_stock_flag': 'Out of Stock',
    'log_reviews': 'Reviews (log)',
    'log_likes': 'Likes (log)',
    'has_retinol': 'Retinol',
    'has_vitamin_c': 'Vitamin C',
    'has_hyaluronic_acid': 'Hyaluronic',
    'has_niacinamide': 'Niacinamide',
    'has_salicylic_acid': 'Salicylic',
    'has_glycolic_acid': 'Glycolic',
    'has_peptides': 'Peptides',
    'has_ceramides': 'Ceramides',
    'has_squalane': 'Squalane',
    'has_collagen': 'Collagen',
    'has_aha_bha': 'AHA/BHA',
    'has_spf': 'SPF',
    'has_fragrance': 'Fragrance',
}

corr_df = df[feature_cols + ['price_tier']].copy()
corr_df.columns = [feature_display_names.get(c, c) for c in feature_cols] + ['Price Tier']
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, annot_kws={'size': 8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/eda_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_correlation.png")

# 3f. Top 15 brands by avg price
fig, ax = plt.subplots(figsize=(10, 6))
top_brands = df.groupby('brand_name')['price'].agg(['mean', 'count']).query('count >= 5').sort_values('mean', ascending=True).tail(15)
colors = ['#E74C3C' if m > median_price else '#5DADE2' for m in top_brands['mean']]
ax.barh(top_brands.index, top_brands['mean'], color=colors, edgecolor='white')
ax.axvline(median_price, color='gray', linestyle='--', alpha=0.7, label=f'Median: ${median_price:.0f}')
ax.set_xlabel('Average Price ($)')
ax.set_title('Top 15 Brands by Average Price (min 5 products)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTDIR}/eda_top_brands.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_top_brands.png")
ingredient_flags = [f'has_{k}' for k in key_ingredients.keys()]
ing_by_tier = df.groupby('price_tier')[ingredient_flags].mean().T
# Ensure both tiers exist
if 0 not in ing_by_tier.columns:
    ing_by_tier[0] = 0.0
if 1 not in ing_by_tier.columns:
    ing_by_tier[1] = 0.0
ing_by_tier = ing_by_tier[[0, 1]]
ing_by_tier.columns = ['Budget', 'Premium']
ing_by_tier.index = [feature_display_names.get(c, c) for c in ingredient_flags]
ing_by_tier = ing_by_tier.sort_values('Premium', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(ing_by_tier))
ax.barh(y_pos - 0.2, ing_by_tier['Budget'], 0.4, label='Budget', color='#5DADE2', alpha=0.85)
ax.barh(y_pos + 0.2, ing_by_tier['Premium'], 0.4, label='Premium', color='#E74C3C', alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(ing_by_tier.index)
ax.set_xlabel('Proportion of Products')
ax.set_title('Ingredient Prevalence: Budget vs Premium')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTDIR}/eda_ingredient_prevalence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_ingredient_prevalence.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════════

X = df[feature_cols].copy()
y = df['price_tier']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale for linear models and MLP
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']

cv_results = {}
fitted_models = {}
test_results = {}
best_params = {}

# ── Helper to evaluate on test set ──
def evaluate_test(model, X_te, y_te, name):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_te, y_prob)
    return {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': (y_pred == y_te).mean(),
        'roc_auc': roc_auc_score(y_te, y_prob),
        'fpr': fpr,
        'tpr': tpr,
        'prec_curve': prec_curve,
        'rec_curve': rec_curve,
        'report': classification_report(y_te, y_pred,
                    target_names=['Budget', 'Premium'], output_dict=True),
        'cm': confusion_matrix(y_te, y_pred),
    }

# ── Helper to extract CV results from cross_validate ──
def extract_cv(cv_res):
    return {
        'accuracy_mean': cv_res['test_accuracy'].mean(),
        'accuracy_std': cv_res['test_accuracy'].std(),
        'f1_mean': cv_res['test_f1'].mean(),
        'f1_std': cv_res['test_f1'].std(),
        'roc_auc_mean': cv_res['test_roc_auc'].mean(),
        'roc_auc_std': cv_res['test_roc_auc'].std(),
        'precision_mean': cv_res['test_precision'].mean(),
        'precision_std': cv_res['test_precision'].std(),
        'recall_mean': cv_res['test_recall'].mean(),
        'recall_std': cv_res['test_recall'].std(),
        'train_accuracy_mean': cv_res['train_accuracy'].mean(),
    }

# ─────────────────────────────────────────────────────
# 4a. Logistic Lasso (L1) — Baseline
# ─────────────────────────────────────────────────────
print("\nTraining: Logistic (Lasso/L1)")
lasso = LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=5000, random_state=42)
cv_res = cross_validate(lasso, X_train_scaled, y_train, cv=cv, scoring=scoring, return_train_score=True)
cv_results['Logistic (Lasso/L1)'] = extract_cv(cv_res)
best_params['Logistic (Lasso/L1)'] = {'penalty': 'l1', 'C': 1.0, 'solver': 'saga'}
lasso.fit(X_train_scaled, y_train)
fitted_models['Logistic (Lasso/L1)'] = lasso
test_results['Logistic (Lasso/L1)'] = evaluate_test(lasso, X_test_scaled, y_test, 'Lasso')
print(f"  CV Acc: {cv_results['Logistic (Lasso/L1)']['accuracy_mean']:.3f}")

# ─────────────────────────────────────────────────────
# 4b. Logistic Ridge (L2) — Baseline
# ─────────────────────────────────────────────────────
print("\nTraining: Logistic (Ridge/L2)")
ridge = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=5000, random_state=42)
cv_res = cross_validate(ridge, X_train_scaled, y_train, cv=cv, scoring=scoring, return_train_score=True)
cv_results['Logistic (Ridge/L2)'] = extract_cv(cv_res)
best_params['Logistic (Ridge/L2)'] = {'penalty': 'l2', 'C': 1.0, 'solver': 'lbfgs'}
ridge.fit(X_train_scaled, y_train)
fitted_models['Logistic (Ridge/L2)'] = ridge
test_results['Logistic (Ridge/L2)'] = evaluate_test(ridge, X_test_scaled, y_test, 'Ridge')
print(f"  CV Acc: {cv_results['Logistic (Ridge/L2)']['accuracy_mean']:.3f}")

# ─────────────────────────────────────────────────────
# 4c. Decision Tree (CART) — GridSearchCV
# ─────────────────────────────────────────────────────
print("\nGridSearchCV: Decision Tree (CART)")
cart_param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [5, 10, 20, 50],
}
cart_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    cart_param_grid, cv=cv, scoring='roc_auc', refit=True, return_train_score=True, n_jobs=2
)
cart_grid.fit(X_train, y_train)
best_params['Decision Tree (CART)'] = cart_grid.best_params_
print(f"  Best params: {cart_grid.best_params_}")

# Get full CV metrics with best params
best_cart = cart_grid.best_estimator_
cv_res = cross_validate(best_cart, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
cv_results['Decision Tree (CART)'] = extract_cv(cv_res)
fitted_models['Decision Tree (CART)'] = best_cart
test_results['Decision Tree (CART)'] = evaluate_test(best_cart, X_test, y_test, 'CART')
print(f"  CV Acc: {cv_results['Decision Tree (CART)']['accuracy_mean']:.3f}")

# ─────────────────────────────────────────────────────
# 4d. Random Forest — GridSearchCV
# ─────────────────────────────────────────────────────
print("\nGridSearchCV: Random Forest")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 8],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, cv=cv, scoring='roc_auc', refit=True, return_train_score=True, n_jobs=2
)
rf_grid.fit(X_train, y_train)
best_params['Random Forest'] = rf_grid.best_params_
print(f"  Best params: {rf_grid.best_params_}")

best_rf = rf_grid.best_estimator_
cv_res = cross_validate(best_rf, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
cv_results['Random Forest'] = extract_cv(cv_res)
fitted_models['Random Forest'] = best_rf
test_results['Random Forest'] = evaluate_test(best_rf, X_test, y_test, 'RF')
print(f"  CV Acc: {cv_results['Random Forest']['accuracy_mean']:.3f}")

# ─────────────────────────────────────────────────────
# 4e. LightGBM — GridSearchCV
# ─────────────────────────────────────────────────────
print("\nGridSearchCV: LightGBM")
lgbm_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
}
lgbm_grid = GridSearchCV(
    lgb.LGBMClassifier(subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=1),
    lgbm_param_grid, cv=cv, scoring='roc_auc', refit=True, return_train_score=True, n_jobs=2
)
lgbm_grid.fit(X_train, y_train)
best_params['LightGBM'] = lgbm_grid.best_params_
print(f"  Best params: {lgbm_grid.best_params_}")

best_lgbm = lgbm_grid.best_estimator_
cv_res = cross_validate(best_lgbm, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
cv_results['LightGBM'] = extract_cv(cv_res)
fitted_models['LightGBM'] = best_lgbm
test_results['LightGBM'] = evaluate_test(best_lgbm, X_test, y_test, 'LGBM')
print(f"  CV Acc: {cv_results['LightGBM']['accuracy_mean']:.3f}")

# ─────────────────────────────────────────────────────
# 4f. XGBoost — GridSearchCV
# ─────────────────────────────────────────────────────
print("\nGridSearchCV: XGBoost")
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
}
xgb_grid = GridSearchCV(
    xgb.XGBClassifier(subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', n_jobs=1),
    xgb_param_grid, cv=cv, scoring='roc_auc', refit=True, return_train_score=True, n_jobs=2
)
xgb_grid.fit(X_train, y_train)
best_params['XGBoost'] = xgb_grid.best_params_
print(f"  Best params: {xgb_grid.best_params_}")

best_xgb = xgb_grid.best_estimator_
cv_res = cross_validate(best_xgb, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
cv_results['XGBoost'] = extract_cv(cv_res)
fitted_models['XGBoost'] = best_xgb
test_results['XGBoost'] = evaluate_test(best_xgb, X_test, y_test, 'XGB')
print(f"  CV Acc: {cv_results['XGBoost']['accuracy_mean']:.3f}")

# ─────────────────────────────────────────────────────
# 4g. Neural Network (MLP) — Keras
# ─────────────────────────────────────────────────────
print("\nTraining: Neural Network (MLP)")

tf.random.set_seed(42)
np.random.seed(42)

mlp_model = keras.Sequential([
    keras.layers.Input(shape=(len(feature_cols),)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])
mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

mlp_history = mlp_model.fit(
    X_train_scaled.values, y_train.values,
    epochs=150, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)
print(f"  Trained for {len(mlp_history.history['loss'])} epochs")

# MLP test evaluation
mlp_y_prob = mlp_model.predict(X_test_scaled.values, verbose=0).flatten()
mlp_y_pred = (mlp_y_prob > 0.5).astype(int)
mlp_fpr, mlp_tpr, _ = roc_curve(y_test, mlp_y_prob)
mlp_prec_c, mlp_rec_c, _ = precision_recall_curve(y_test, mlp_y_prob)

test_results['Neural Network (MLP)'] = {
    'y_pred': mlp_y_pred,
    'y_prob': mlp_y_prob,
    'accuracy': (mlp_y_pred == y_test.values).mean(),
    'roc_auc': roc_auc_score(y_test, mlp_y_prob),
    'fpr': mlp_fpr,
    'tpr': mlp_tpr,
    'prec_curve': mlp_prec_c,
    'rec_curve': mlp_rec_c,
    'report': classification_report(y_test, mlp_y_pred,
                target_names=['Budget', 'Premium'], output_dict=True),
    'cm': confusion_matrix(y_test, mlp_y_pred),
}

# MLP CV via manual K-fold (Keras doesn't support sklearn cross_validate directly)
mlp_cv_acc, mlp_cv_f1, mlp_cv_auc = [], [], []
from sklearn.metrics import f1_score
for train_idx, val_idx in cv.split(X_train_scaled, y_train):
    Xc_tr, Xc_val = X_train_scaled.values[train_idx], X_train_scaled.values[val_idx]
    yc_tr, yc_val = y_train.values[train_idx], y_train.values[val_idx]

    m = keras.Sequential([
        keras.layers.Input(shape=(len(feature_cols),)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.fit(Xc_tr, yc_tr, epochs=100, batch_size=32, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)])

    probs = m.predict(Xc_val, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)
    mlp_cv_acc.append((preds == yc_val).mean())
    mlp_cv_f1.append(f1_score(yc_val, preds))
    mlp_cv_auc.append(roc_auc_score(yc_val, probs))

cv_results['Neural Network (MLP)'] = {
    'accuracy_mean': np.mean(mlp_cv_acc),
    'accuracy_std': np.std(mlp_cv_acc),
    'f1_mean': np.mean(mlp_cv_f1),
    'f1_std': np.std(mlp_cv_f1),
    'roc_auc_mean': np.mean(mlp_cv_auc),
    'roc_auc_std': np.std(mlp_cv_auc),
    'precision_mean': 0.0,  # placeholder
    'precision_std': 0.0,
    'recall_mean': 0.0,
    'recall_std': 0.0,
    'train_accuracy_mean': 0.0,
}
best_params['Neural Network (MLP)'] = {
    'hidden_layers': '128-128', 'dropout': 0.3,
    'optimizer': 'adam', 'epochs': len(mlp_history.history['loss']),
}
print(f"  CV Acc: {np.mean(mlp_cv_acc):.3f}, CV AUC: {np.mean(mlp_cv_auc):.3f}")
print(f"  Test Acc: {test_results['Neural Network (MLP)']['accuracy']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# 5a. CV comparison bar chart
cv_df = pd.DataFrame(cv_results).T
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(cv_df))
width = 0.25
ax.bar(x - width, cv_df['accuracy_mean'], width, yerr=cv_df['accuracy_std'],
       label='Accuracy', color='#3498DB', capsize=3, alpha=0.85)
ax.bar(x, cv_df['f1_mean'], width, yerr=cv_df['f1_std'],
       label='F1 Score', color='#E74C3C', capsize=3, alpha=0.85)
ax.bar(x + width, cv_df['roc_auc_mean'], width, yerr=cv_df['roc_auc_std'],
       label='ROC-AUC', color='#2ECC71', capsize=3, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(cv_df.index, rotation=20, ha='right')
ax.set_ylabel('Score')
ax.set_title('5-Fold Cross-Validation: Model Comparison', fontsize=14, fontweight='bold')
ax.legend()
ax.set_ylim(0.5, 1.0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/model_cv_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: model_cv_comparison.png")

# 5b. ROC curves
fig, ax = plt.subplots(figsize=(8, 7))
colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']
for (name, res), color in zip(test_results.items(), colors):
    ax.plot(res['fpr'], res['tpr'], label=f"{name} (AUC={res['roc_auc']:.3f})", color=color, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/model_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_roc_curves.png")

# 5c. Confusion matrices
n_models = len(test_results)
n_cols = 4
n_rows = (n_models + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()
for idx, (name, res) in enumerate(test_results.items()):
    ax = axes[idx]
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Budget', 'Premium'], yticklabels=['Budget', 'Premium'], ax=ax)
    ax.set_title(name, fontweight='bold', fontsize=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
for idx in range(len(test_results), len(axes)):
    axes[idx].set_visible(False)
plt.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/model_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_confusion_matrices.png")

# 5d. MLP Training History
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(mlp_history.history['loss'], label='Train Loss', color='#3498DB')
axes[0].plot(mlp_history.history['val_loss'], label='Val Loss', color='#E74C3C')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('MLP Training Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(mlp_history.history['accuracy'], label='Train Accuracy', color='#3498DB')
axes[1].plot(mlp_history.history['val_accuracy'], label='Val Accuracy', color='#E74C3C')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('MLP Training Accuracy')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Neural Network (MLP) Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/mlp_training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: mlp_training_history.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SHAP ANALYSIS (tree + linear models)
# ═══════════════════════════════════════════════════════════════════════════════

shap_models = {
    'Decision Tree (CART)': ('tree', X_test),
    'Random Forest': ('tree', X_test),
    'LightGBM': ('tree', X_test),
    'XGBoost': ('tree', X_test),
}

shap_results = {}
display_feature_names = [feature_display_names.get(f, f) for f in feature_cols]

for name, (explainer_type, X_shap) in shap_models.items():
    print(f"\nSHAP: {name}")
    explainer = shap.TreeExplainer(fitted_models[name])
    shap_vals = explainer(X_shap)
    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]
    shap_vals.feature_names = display_feature_names
    shap_results[name] = {'explainer': explainer, 'shap_values': shap_vals}

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_vals, show=False, max_display=15)
    plt.title(f"SHAP Beeswarm — {name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    safe_name = name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    plt.savefig(f'{OUTDIR}/shap_beeswarm_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_vals, show=False, max_display=15)
    plt.title(f"Feature Importance — {name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/shap_bar_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

print("\nAll SHAP (tree) plots saved.")

# Linear model SHAP
for name in ['Logistic (Lasso/L1)', 'Logistic (Ridge/L2)']:
    print(f"\nSHAP: {name}")
    m = fitted_models[name]
    explainer = shap.LinearExplainer(m, X_train_scaled)
    shap_vals = explainer(X_test_scaled)
    shap_vals.feature_names = display_feature_names
    shap_results[name] = {'explainer': explainer, 'shap_values': shap_vals}

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_vals, show=False, max_display=15)
    plt.title(f"SHAP Beeswarm — {name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    safe_name = name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    plt.savefig(f'{OUTDIR}/shap_beeswarm_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_vals, show=False, max_display=15)
    plt.title(f"Feature Importance — {name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/shap_bar_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

print("All SHAP (linear) plots saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAVE ALL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

artifacts = {
    'fitted_models': fitted_models,
    'cv_results': cv_results,
    'test_results': {k: {kk: vv for kk, vv in v.items() if kk not in ('fpr','tpr','prec_curve','rec_curve')}
                     for k, v in test_results.items()},
    'shap_results': shap_results,
    'feature_cols': feature_cols,
    'feature_display_names': feature_display_names,
    'le_cat': le_cat,
    'scaler': scaler,
    'median_price': median_price,
    'key_ingredients': key_ingredients,
    'luxury_brands': luxury_brands,
    'main_categories': main_categories,
    'best_params': best_params,
    'brand_counts': brand_counts,
    'X_train': X_train,
    'X_test': X_test,
    'X_train_scaled': X_train_scaled,
    'X_test_scaled': X_test_scaled,
    'y_train': y_train,
    'y_test': y_test,
    'mlp_history': {k: v for k, v in mlp_history.history.items()},
    'n_products': len(df),
    'budget_count': int((y == 0).sum()),
    'premium_count': int((y == 1).sum()),
}

with open(f'{OUTDIR}/model_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# Save Keras model separately
mlp_model.save(f'{OUTDIR}/mlp_model.keras')

print(f"\nAll artifacts saved to {OUTDIR}/model_artifacts.pkl")
print(f"MLP model saved to {OUTDIR}/mlp_model.keras")
print("Pipeline complete!")
