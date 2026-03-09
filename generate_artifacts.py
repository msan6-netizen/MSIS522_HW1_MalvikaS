"""
generate_artifacts.py
---------------------
Generates model_artifacts.pkl in the same directory as this script.

Run once before launching the Streamlit app:
    python generate_artifacts.py

The script builds a small sample Sephora-style product dataset and computes
a cosine-similarity matrix from TF-IDF ingredient vectors so the dashboard
has something to work with out of the box.  Replace the sample data with the
real dataset to get production-quality recommendations.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Sample product data (replace with real Sephora dataset)
# ---------------------------------------------------------------------------
SAMPLE_PRODUCTS = [
    {
        "product_name": "Hydrating Facial Serum",
        "category": "Skincare",
        "rating": 4.5,
        "price": 48.00,
        "ingredients": "hyaluronic acid niacinamide glycerin panthenol ceramide",
    },
    {
        "product_name": "Vitamin C Brightening Serum",
        "category": "Skincare",
        "rating": 4.3,
        "price": 52.00,
        "ingredients": "ascorbic acid vitamin c ferulic acid hyaluronic acid glycerin",
    },
    {
        "product_name": "Retinol Night Cream",
        "category": "Skincare",
        "rating": 4.6,
        "price": 65.00,
        "ingredients": "retinol peptides ceramide niacinamide hyaluronic acid",
    },
    {
        "product_name": "SPF 50 Sunscreen",
        "category": "Skincare",
        "rating": 4.7,
        "price": 34.00,
        "ingredients": "zinc oxide titanium dioxide hyaluronic acid glycerin niacinamide",
    },
    {
        "product_name": "Gentle Exfoliating Toner",
        "category": "Skincare",
        "rating": 4.2,
        "price": 29.00,
        "ingredients": "glycolic acid lactic acid hyaluronic acid aloe vera panthenol",
    },
    {
        "product_name": "Matte Foundation",
        "category": "Makeup",
        "rating": 4.4,
        "price": 42.00,
        "ingredients": "dimethicone silica kaolin talc glycerin",
    },
    {
        "product_name": "Dewy Foundation",
        "category": "Makeup",
        "rating": 4.1,
        "price": 38.00,
        "ingredients": "glycerin hyaluronic acid dimethicone niacinamide",
    },
    {
        "product_name": "Long-Wear Mascara",
        "category": "Makeup",
        "rating": 4.5,
        "price": 24.00,
        "ingredients": "beeswax carnauba wax glycerin panthenol vitamin e",
    },
    {
        "product_name": "Volumizing Mascara",
        "category": "Makeup",
        "rating": 4.3,
        "price": 22.00,
        "ingredients": "beeswax carnauba wax dimethicone glycerin panthenol",
    },
    {
        "product_name": "Lip Plumping Gloss",
        "category": "Makeup",
        "rating": 4.0,
        "price": 18.00,
        "ingredients": "castor oil hyaluronic acid vitamin e glycerin peptides",
    },
    {
        "product_name": "Moisturizing Shampoo",
        "category": "Hair",
        "rating": 4.4,
        "price": 28.00,
        "ingredients": "argan oil keratin panthenol glycerin biotin",
    },
    {
        "product_name": "Repairing Hair Mask",
        "category": "Hair",
        "rating": 4.6,
        "price": 35.00,
        "ingredients": "keratin argan oil coconut oil panthenol biotin glycerin",
    },
    {
        "product_name": "Volumizing Conditioner",
        "category": "Hair",
        "rating": 4.2,
        "price": 26.00,
        "ingredients": "panthenol biotin glycerin wheat protein silicone",
    },
    {
        "product_name": "Floral Eau de Parfum",
        "category": "Fragrance",
        "rating": 4.7,
        "price": 120.00,
        "ingredients": "rose jasmine sandalwood musk amber",
    },
    {
        "product_name": "Woody Cologne",
        "category": "Fragrance",
        "rating": 4.5,
        "price": 95.00,
        "ingredients": "cedarwood sandalwood vetiver musk amber bergamot",
    },
]


def build_artifacts(products_data: list[dict]) -> dict:
    products = pd.DataFrame(products_data)

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(products["ingredients"].fillna(""))
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return {
        "products": products,
        "similarity_matrix": sim_matrix,
        "tfidf_vectorizer": tfidf,
    }


def main():
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts.pkl")

    print("Building artifacts …")
    artifacts = build_artifacts(SAMPLE_PRODUCTS)

    with open(output_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Saved → {output_path}")
    print(f"  Products : {len(artifacts['products'])}")
    print(f"  Matrix   : {artifacts['similarity_matrix'].shape}")


if __name__ == "__main__":
    main()
