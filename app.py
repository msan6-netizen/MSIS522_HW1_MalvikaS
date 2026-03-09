import os
import pickle
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Artifact loading – path is relative to this file so the app works on any
# machine regardless of where the repo is checked out.
# ---------------------------------------------------------------------------
ARTIFACTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts.pkl")


@st.cache_resource
def load_artifacts():
    if not os.path.exists(ARTIFACTS_PATH):
        st.error(
            f"model_artifacts.pkl not found at `{ARTIFACTS_PATH}`. "
            "Please run `python generate_artifacts.py` first."
        )
        st.stop()
    with open(ARTIFACTS_PATH, "rb") as f:
        art = pickle.load(f)
    return art


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sephora Product Recommender",
    page_icon="💄",
    layout="wide",
)

st.title("💄 Sephora Product Recommender")
st.markdown("Discover products similar to your favourites based on ingredients and ratings.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
art = load_artifacts()

products: pd.DataFrame = art["products"]
similarity_matrix = art["similarity_matrix"]
tfidf_vectorizer = art["tfidf_vectorizer"]

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

categories = ["All"] + sorted(products["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Category", categories)

min_rating, max_rating = float(products["rating"].min()), float(products["rating"].max())
rating_range = st.sidebar.slider(
    "Minimum Rating",
    min_value=min_rating,
    max_value=max_rating,
    value=min_rating,
    step=0.1,
)

# ---------------------------------------------------------------------------
# Product selector
# ---------------------------------------------------------------------------
filtered = products.copy()
if selected_category != "All":
    filtered = filtered[filtered["category"] == selected_category]
filtered = filtered[filtered["rating"] >= rating_range]

if filtered.empty:
    st.warning("No products match the current filters.")
    st.stop()

selected_product = st.selectbox(
    "Select a product to get recommendations:",
    options=filtered["product_name"].tolist(),
)

# ---------------------------------------------------------------------------
# Display selected product details
# ---------------------------------------------------------------------------
product_row = products[products["product_name"] == selected_product].iloc[0]

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Rating", f"{product_row['rating']:.1f} / 5.0")
    st.metric("Price", f"${product_row['price']:.2f}")
    st.metric("Category", product_row["category"])
with col2:
    st.subheader("Key Ingredients")
    st.write(product_row.get("ingredients", "N/A"))

st.divider()

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
st.subheader("Similar Products")

n_recs = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

product_idx = products[products["product_name"] == selected_product].index[0]
sim_scores = list(enumerate(similarity_matrix[product_idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
# Exclude the selected product itself
sim_scores = [(i, s) for i, s in sim_scores if i != product_idx][:n_recs]

rec_indices = [i for i, _ in sim_scores]
rec_scores = [s for _, s in sim_scores]

recommendations = products.iloc[rec_indices][
    ["product_name", "category", "rating", "price"]
].copy()
recommendations.insert(0, "Similarity Score", [round(s, 3) for s in rec_scores])
recommendations = recommendations.reset_index(drop=True)
recommendations.index += 1

st.dataframe(recommendations, use_container_width=True)

# ---------------------------------------------------------------------------
# Category distribution chart
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Category Distribution in Current Filter")
cat_counts = filtered["category"].value_counts()
st.bar_chart(cat_counts)
