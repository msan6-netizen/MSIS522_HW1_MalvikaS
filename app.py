"""
Sephora Premium vs Budget Price Tier — Full Dashboard
=====================================================
Streamlit app with Executive Summary, Descriptive Analytics,
Model Performance, and Explainability & Interactive Prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Sephora Price Tier Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load artifacts ──────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    with open(r'C:\Malvika\sephora_dashboard\model_artifacts.pkl', 'rb') as f:
        return pickle.load(f)

art = load_artifacts()
fitted_models = art['fitted_models']
cv_results = art['cv_results']
test_results = art['test_results']
shap_results = art['shap_results']
feature_cols = art['feature_cols']
display_names = art['feature_display_names']
le_cat = art['le_cat']
scaler = art['scaler']
median_price = art['median_price']
key_ingredients = art['key_ingredients']
luxury_brands = art['luxury_brands']
main_categories = art['main_categories']
best_params = art['best_params']
n_products = art['n_products']
budget_count = art['budget_count']
premium_count = art['premium_count']
mlp_history = art.get('mlp_history', None)
OUTDIR = r'C:\Malvika\sephora_dashboard'

# Compute dynamic metrics
best_model = max(cv_results, key=lambda k: cv_results[k]['roc_auc_mean'])
best_auc = cv_results[best_model]['roc_auc_mean']
best_acc = cv_results[best_model]['accuracy_mean']

# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("Sephora Price Tier Analysis")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {n_products} Sephora products")
st.sidebar.markdown(f"**Price Threshold:** > EUR{median_price:.0f} = Premium")
st.sidebar.markdown(f"**Target Split:** {budget_count} Budget / {premium_count} Premium")
st.sidebar.markdown("**Models:** 7 (Lasso, Ridge, CART, RF, LGBM, XGB, MLP)")
st.sidebar.markdown(f"**Best Model:** {best_model}")
st.sidebar.markdown(f"**Best CV AUC:** {best_auc:.3f}")

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Interactive Prediction",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.title("Executive Summary")
    st.markdown("### What Justifies a Premium Price at Sephora?")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Products Analyzed", f"{n_products}")
    col2.metric("Best Model", best_model.split('(')[0].strip())
    col3.metric("Best ROC-AUC", f"{best_auc:.3f}")
    col4.metric("Best CV Accuracy", f"{best_acc:.1%}")

    st.markdown("---")

    st.markdown(f"""
    ### About the Dataset

    This dataset contains **{n_products} products** scraped from the Sephora France website,
    spanning five major categories: Parfum, Maquillage (Makeup), Soin Visage (Skincare),
    Cheveux (Hair), and Corps & Bain (Body & Bath). Each product record includes the
    listing price, brand information, customer ratings, stock status, and a full ingredient
    list. Prices range from around EUR5 to nearly EUR1,000, with a median of **EUR{median_price:.0f}**.
    We engineered **{len(feature_cols)} features** from this raw data, including ingredient counts,
    binary flags for 13 key active ingredients (retinol, vitamin C, hyaluronic acid, peptides,
    fragrance, etc.), brand popularity metrics, and a luxury-brand indicator.

    ### The Prediction Task

    The target variable is a binary **Price Tier** label: products priced above the
    EUR{median_price:.0f} median are classified as **Premium** ({premium_count} products), while those
    at or below the median are **Budget** ({budget_count} products). This near-balanced split
    (approximately 50/50) means we do not need special handling for class imbalance.
    Predicting price tier is valuable because it reveals the attributes that *justify*
    higher prices in the eyes of consumers and retailers — information that is directly
    actionable for brand positioning, product development, and assortment strategy.

    ### Why This Matters

    Understanding what drives premium pricing in the beauty industry is critical for
    multiple stakeholders. **Brand managers** can use these insights to decide which
    ingredient profiles and brand positioning strategies justify a higher price point.
    **Retailers** can optimize their product mix by understanding which product attributes
    correlate with premium pricing. **Consumers** benefit from transparency — knowing
    whether they are paying for formulation complexity, brand prestige, or specific
    active ingredients.

    ### Approach and Key Findings

    We built and compared **7 classification models**: two linear baselines (Logistic Lasso
    and Ridge), a Decision Tree, Random Forest, LightGBM, XGBoost (all tuned via
    GridSearchCV), and a Keras Neural Network (MLP). All models were evaluated using
    5-fold stratified cross-validation and a 20% held-out test set.

    The **{best_model}** achieved the highest cross-validated ROC-AUC of **{best_auc:.3f}**.
    SHAP analysis revealed that **product category** (especially Parfum) is the single
    strongest price driver, followed by **formulation complexity** (ingredient count) and
    **luxury brand status**. Individual "hero" ingredients like retinol or vitamin C have
    surprisingly modest effects on their own — it is the *combination* of category,
    complexity, and brand prestige that most reliably predicts premium pricing.

    ### Key Findings

    | # | Finding | Business Implication |
    |---|---------|---------------------|
    | 1 | **Product category is the #1 price driver** — Parfum dominates Premium | Category positioning matters more than individual ingredients |
    | 2 | **Formulation complexity (ingredient count) ranks #2** | More ingredients = perceived higher value |
    | 3 | **Luxury brand status has a strong but not dominant effect** | Brand alone doesn't guarantee premium |
    | 4 | **Fragrance in the formula strongly correlates with premium** | Perfumed products command higher prices |
    | 5 | **"Hero" ingredients have modest individual effects** | Marketing single ingredients alone doesn't justify premium pricing |
    | 6 | **Ensemble tree models perform best** | Non-linear feature interactions matter |
    """)

    st.markdown("---")
    st.markdown("### Cross-Validation Results at a Glance")
    st.image(f'{OUTDIR}/model_cv_comparison.png', use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: DESCRIPTIVE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.title("Descriptive Analytics")
    st.markdown("Understanding the Sephora product landscape before modeling.")

    st.markdown("---")
    st.markdown("### Price Distribution & Target Variable")
    st.image(f'{OUTDIR}/eda_price_dist.png', use_container_width=True)
    st.markdown(f"""
    The price distribution is **right-skewed** with a median of EUR{median_price:.0f}. Most products
    cluster between EUR10 and EUR80, with a long tail of luxury items reaching EUR900+. We split
    at the median to create a balanced binary target: {budget_count} Budget vs {premium_count}
    Premium products.
    """)

    st.markdown("---")
    st.markdown("### Price by Product Category")
    st.image(f'{OUTDIR}/eda_price_by_category.png', use_container_width=True)
    st.markdown("""
    **Parfum** has the highest median price and widest spread, followed by Soin Visage (Skincare).
    Corps & Bain (Body & Bath) and Cheveux (Hair) skew Budget. This category-level price
    separation explains why category is the strongest single predictor in our models.
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Luxury vs Non-Luxury Brands")
        st.image(f'{OUTDIR}/eda_luxury_vs_nonluxury.png', use_container_width=True)
        st.markdown("""
        Luxury brands (Dior, Tom Ford, Guerlain, etc.) have a significantly higher median price
        than non-luxury brands. However, substantial overlap exists — not all luxury brand
        products are Premium, and some non-luxury brands reach high price points.
        """)
    with col2:
        st.markdown("### Ingredient Complexity by Tier")
        st.image(f'{OUTDIR}/eda_ingredients_by_tier.png', use_container_width=True)
        st.markdown("""
        Premium products tend to have more ingredients on average, as shown by the wider and
        higher distribution of the Premium violin. This suggests formulation complexity
        is associated with higher prices.
        """)

    st.markdown("---")
    st.markdown("### Ingredient Prevalence: Budget vs Premium")
    st.image(f'{OUTDIR}/eda_ingredient_prevalence.png', use_container_width=True)
    st.markdown("""
    **Fragrance** stands out as far more prevalent in Premium products than Budget ones. Other
    active ingredients (peptides, vitamin C, hyaluronic acid) show slight Premium skew, but
    the differences are modest — suggesting no single ingredient is a "silver bullet" for
    premium pricing.
    """)

    st.markdown("---")
    st.markdown("### Top 15 Brands by Average Price")
    st.image(f'{OUTDIR}/eda_top_brands.png', use_container_width=True)
    st.markdown(f"""
    Among brands with at least 5 products, the luxury houses (Tom Ford, Dior, Guerlain) dominate
    the top of the price ranking. The dashed line marks the EUR{median_price:.0f} median — brands
    above it are predominantly Premium.
    """)

    st.markdown("---")
    st.markdown("### Feature Correlation Heatmap")
    st.image(f'{OUTDIR}/eda_correlation.png', use_container_width=True)
    st.markdown("""
    Notable correlations with Price Tier: **Category** shows the strongest single correlation,
    followed by **Ingredient Count** and **Luxury Brand**. The low multicollinearity among
    features (most off-diagonal values near zero) is good for modeling — our features capture
    distinct dimensions of a product.
    """)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.title("Model Performance")
    st.markdown("Seven models trained with 5-fold stratified cross-validation. "
                "Decision Tree, Random Forest, LightGBM, and XGBoost were tuned via GridSearchCV.")

    st.markdown("---")
    st.markdown("### Cross-Validation Performance")
    st.image(f'{OUTDIR}/model_cv_comparison.png', use_container_width=True)

    # CV results table
    cv_df = pd.DataFrame(cv_results).T
    display_cv = pd.DataFrame({
        'Accuracy': cv_df.apply(lambda r: f"{r['accuracy_mean']:.3f} +/- {r['accuracy_std']:.3f}", axis=1),
        'F1 Score': cv_df.apply(lambda r: f"{r['f1_mean']:.3f} +/- {r['f1_std']:.3f}", axis=1),
        'ROC-AUC': cv_df.apply(lambda r: f"{r['roc_auc_mean']:.3f} +/- {r['roc_auc_std']:.3f}", axis=1),
    })
    st.dataframe(display_cv, use_container_width=True)

    st.markdown(f"""
    **Observations:**
    - **{best_model}** achieves the highest CV ROC-AUC ({best_auc:.3f})
    - Ensemble models (Random Forest, LightGBM, XGBoost) generally outperform linear baselines
    - The Neural Network (MLP) is competitive, showing that the feature space supports non-linear learning
    - The gap between train and test accuracy is small across all models — no severe overfitting
    """)

    # Best Hyperparameters
    st.markdown("---")
    st.markdown("### Best Hyperparameters (from GridSearchCV)")
    for model_name, params in best_params.items():
        st.markdown(f"**{model_name}:** `{params}`")

    st.markdown("---")
    st.markdown("### ROC Curves")
    st.image(f'{OUTDIR}/model_roc_curves.png', use_container_width=True)
    st.markdown("""
    All models are well above the diagonal (random baseline), confirming that the features
    carry real predictive signal. Ensemble tree models and the MLP cluster near the top.
    """)

    st.markdown("---")
    st.markdown("### Confusion Matrices")
    st.image(f'{OUTDIR}/model_confusion_matrices.png', use_container_width=True)

    # MLP Training History
    if mlp_history:
        st.markdown("---")
        st.markdown("### Neural Network (MLP) Training History")
        st.image(f'{OUTDIR}/mlp_training_history.png', use_container_width=True)
        st.markdown(f"""
        The MLP was trained with early stopping (patience=15) on validation loss. Training
        converged after **{len(mlp_history['loss'])} epochs**. The training and validation
        curves track closely, indicating good generalization without severe overfitting.
        """)

    st.markdown("---")
    st.markdown("### Detailed Test Set Results")
    model_select = st.selectbox("Select model for detailed report:", list(test_results.keys()))
    res = test_results[model_select]
    report = res['report']

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", f"{res['accuracy']:.1%}")
    col2.metric("Test ROC-AUC", f"{res['roc_auc']:.4f}")
    col3.metric("Test F1 (macro)", f"{report['macro avg']['f1-score']:.3f}")

    report_df = pd.DataFrame(report).T.drop('accuracy', errors='ignore')
    report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
    st.dataframe(report_df.style.format({
        'precision': '{:.3f}', 'recall': '{:.3f}', 'f1-score': '{:.3f}', 'support': '{:.0f}'
    }), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: EXPLAINABILITY & INTERACTIVE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.title("Explainability & Interactive Prediction")

    # ── SHAP Section ──
    st.markdown("## SHAP Analysis")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** decomposes each prediction into feature contributions.
    This reveals *why* each model classifies a product as Premium or Budget.
    """)

    shap_model = st.selectbox("Select model for SHAP analysis:", list(shap_results.keys()))
    safe_name = shap_model.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"### Feature Importance — {shap_model}")
        st.image(f'{OUTDIR}/shap_bar_{safe_name}.png', use_container_width=True)
        st.markdown("""
        The bar chart shows **mean absolute SHAP values** — the average magnitude of each feature's
        contribution to predictions. Higher values mean the feature is more important overall.
        """)
    with col_s2:
        st.markdown(f"### SHAP Beeswarm — {shap_model}")
        st.image(f'{OUTDIR}/shap_beeswarm_{safe_name}.png', use_container_width=True)
        st.markdown("""
        Each dot is one product. **Red** = high feature value, **Blue** = low. Position right
        of center pushes toward Premium; left pushes toward Budget.
        """)

    st.markdown("""
    ---
    ### SHAP Interpretation

    **Which features have the strongest impact?** Across tree-based models, **Category** consistently
    ranks #1, followed by **Ingredient Count** and **Brand Popularity / Luxury Brand**. For linear
    models, the ranking shifts slightly but the top features remain similar.

    **How do they influence predictions?** High category values (Parfum) strongly push toward Premium.
    More ingredients push toward Premium. Being a luxury brand pushes toward Premium. Fragrance
    presence pushes toward Premium.

    **Business implications:** A decision-maker launching a new product should focus on (1) which
    category to position in, (2) formulation complexity, and (3) brand prestige. Adding a single
    "hero" ingredient (e.g., retinol) alone is unlikely to justify a premium price — the overall
    product profile matters far more.
    """)

    # ── Interactive Prediction Section ──
    st.markdown("---")
    st.markdown("## Interactive Prediction")
    st.markdown("""
    Configure a hypothetical product below and see how each model predicts its price tier.
    The SHAP waterfall shows exactly *why* the prediction was made.
    """)

    sim_model_name = st.selectbox("Choose prediction model:", list(fitted_models.keys()), index=3)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Product Basics")
        category = st.selectbox("Category", main_categories)
        brand_pop = st.slider("Brand Popularity (# products on Sephora)", 1, 60, 15)
        is_luxury = st.checkbox("Luxury Brand (Dior, Tom Ford, Guerlain, etc.)")
        reviews = st.slider("Expected Reviews Count", 0, 5000, 100)
        likes = st.slider("Expected Likes Count", 0, 50000, 1000)
        out_of_stock = st.checkbox("Out of stock")

    with col2:
        st.subheader("Ingredient Profile")
        ingredient_count = st.slider("Number of Ingredients", 1, 80, 25)

        ing_col1, ing_col2 = st.columns(2)
        with ing_col1:
            has_retinol = st.checkbox("Retinol")
            has_vitamin_c = st.checkbox("Vitamin C")
            has_hyaluronic = st.checkbox("Hyaluronic Acid")
            has_niacinamide = st.checkbox("Niacinamide")
            has_salicylic = st.checkbox("Salicylic Acid")
            has_glycolic = st.checkbox("Glycolic Acid")
            has_fragrance = st.checkbox("Fragrance/Parfum")
        with ing_col2:
            has_peptides = st.checkbox("Peptides")
            has_ceramides = st.checkbox("Ceramides")
            has_squalane = st.checkbox("Squalane")
            has_collagen = st.checkbox("Collagen")
            has_aha_bha = st.checkbox("AHA/BHA")
            has_spf = st.checkbox("SPF/Sunscreen")

    # Build feature vector
    cat_encoded = le_cat.transform([category])[0]
    features_dict = {
        'category_encoded': cat_encoded,
        'ingredient_count': ingredient_count,
        'brand_frequency': brand_pop,
        'is_luxury_brand': int(is_luxury),
        'out_of_stock_flag': int(out_of_stock),
        'log_reviews': np.log1p(reviews),
        'log_likes': np.log1p(likes),
        'has_retinol': int(has_retinol),
        'has_vitamin_c': int(has_vitamin_c),
        'has_hyaluronic_acid': int(has_hyaluronic),
        'has_niacinamide': int(has_niacinamide),
        'has_salicylic_acid': int(has_salicylic),
        'has_glycolic_acid': int(has_glycolic),
        'has_peptides': int(has_peptides),
        'has_ceramides': int(has_ceramides),
        'has_squalane': int(has_squalane),
        'has_collagen': int(has_collagen),
        'has_aha_bha': int(has_aha_bha),
        'has_spf': int(has_spf),
        'has_fragrance': int(has_fragrance),
    }
    features = pd.DataFrame([features_dict])

    # Scale if linear model
    is_linear = 'Logistic' in sim_model_name
    features_input = pd.DataFrame(scaler.transform(features), columns=feature_cols) if is_linear else features

    st.markdown("---")

    # Predictions from ALL models (excluding MLP which isn't in fitted_models)
    st.markdown("### Predictions Across All Models")
    pred_rows = []
    for name, model in fitted_models.items():
        is_lin = 'Logistic' in name
        f_in = pd.DataFrame(scaler.transform(features), columns=feature_cols) if is_lin else features
        prob = model.predict_proba(f_in)[0]
        pred = model.predict(f_in)[0]
        pred_rows.append({
            'Model': name,
            'Prediction': 'PREMIUM' if pred == 1 else 'BUDGET',
            'Premium Prob.': f"{prob[1]:.1%}",
            'Budget Prob.': f"{prob[0]:.1%}",
        })
    pred_df = pd.DataFrame(pred_rows)
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

    # Highlight selected model
    selected_model = fitted_models[sim_model_name]
    prob = selected_model.predict_proba(features_input)[0]
    pred = selected_model.predict(features_input)[0]

    st.markdown(f"### Selected Model: {sim_model_name}")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prediction", "PREMIUM" if pred == 1 else "BUDGET")
    col_b.metric("Premium Probability", f"{prob[1]:.1%}")
    col_c.metric("Budget Probability", f"{prob[0]:.1%}")

    # SHAP waterfall for selected model
    st.markdown("### SHAP Waterfall: Why This Prediction?")
    if sim_model_name in shap_results:
        try:
            shap_info = shap_results[sim_model_name]
            explainer = shap_info['explainer']
            shap_vals = explainer(features_input)
            if len(shap_vals.shape) == 3:
                shap_vals = shap_vals[:, :, 1]
            shap_vals.feature_names = [display_names.get(f, f) for f in feature_cols]

            fig, ax = plt.subplots(figsize=(10, 7))
            shap.plots.waterfall(shap_vals[0], show=False, max_display=15)
            plt.title(f"SHAP Waterfall — {sim_model_name}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP waterfall not available for {sim_model_name}: {e}")
    else:
        st.info(f"SHAP waterfall is not available for {sim_model_name}.")

    st.markdown("""
    **Reading the waterfall chart:**
    - Each bar shows how a single feature pushes the prediction
    - **Red bars** push toward **Premium**; **Blue bars** push toward **Budget**
    - The base value (E[f(x)]) is the average model output; f(x) is this product's score
    - The longer the bar, the stronger that feature's influence on this specific product
    """)

    st.markdown("---")
    st.markdown("""
    ### Try These Scenarios

    | Scenario | Settings | Expected Result |
    |----------|----------|----------------|
    | **Luxury Parfum** | Category=Parfum, Luxury=Yes, Fragrance=Yes, 40 ingredients | Strong Premium |
    | **Simple Moisturizer** | Category=Soin Visage, 10 ingredients, no luxury | Likely Budget |
    | **Hero Ingredient Skincare** | Category=Soin Visage, Retinol+VitC+Peptides, 30 ingredients | Borderline |
    | **Budget Hair Product** | Category=Cheveux, 15 ingredients, no luxury | Strong Budget |
    """)
