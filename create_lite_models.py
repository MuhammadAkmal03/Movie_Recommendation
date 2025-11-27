"""
Create Lite Models for Deployment
Filters top 5,000 movies by rating count and creates optimized models
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import os

print("="*70)
print("CREATING LITE MODELS FOR DEPLOYMENT")
print("="*70)

# Create output directory
os.makedirs("models_lite", exist_ok=True)

# Step 1: Load original models
print("\n1. Loading original models...")
with open("models/movies_tag.pkl", "rb") as f:
    movies_tag = pickle.load(f)
with open("models/org_dataset.pkl", "rb") as f:
    org_dataset = pickle.load(f)
with open("models/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)
with open("models/movie_user_matrix.pkl", "rb") as f:
    movie_user_matrix = pickle.load(f)

print(f" Loaded {len(movies_tag)} movies")
print(f" Movie-user matrix shape: {movie_user_matrix.shape}")

# Step 2: Identify top 5,000 movies by rating count
print("\n2. Identifying top 5,000 movies by rating count...")

# Count ratings per movie
rating_counts = movie_user_matrix.count(axis=1).sort_values(ascending=False)
top_5000_ids = rating_counts.head(5000).index.tolist()

print(f"Selected top 5,000 movies")
print(f"Most rated movie: {rating_counts.iloc[0]:.0f} ratings")
print(f"5,000th movie: {rating_counts.iloc[4999]:.0f} ratings")

# Step 3: Filter movies_tag
print("\n3. Filtering movies_tag...")
movies_tag_lite = movies_tag[movies_tag['id'].isin(top_5000_ids)].copy()
movies_tag_lite = movies_tag_lite.reset_index(drop=True)
print(f"Filtered to {len(movies_tag_lite)} movies")

# Step 4: Filter org_dataset
print("\n4. Filtering org_dataset...")
org_dataset_lite = org_dataset[org_dataset['id'].isin(top_5000_ids)].copy()
org_dataset_lite = org_dataset_lite.reset_index(drop=True)
print(f"Filtered to {len(org_dataset_lite)} movies")

# Step 5: Filter tfidf_matrix
print("\n5. Filtering tfidf_matrix...")
# Get indices of movies in movies_tag that are in top 5000
mask = movies_tag['id'].isin(top_5000_ids)
indices = movies_tag[mask].index.tolist()
tfidf_matrix_lite = tfidf_matrix[indices]
print(f" Filtered to shape: {tfidf_matrix_lite.shape}")

# Step 6: Filter movie_user_matrix
print("\n6. Filtering movie_user_matrix...")
movie_user_matrix_lite = movie_user_matrix.loc[top_5000_ids].copy()
print(f" Filtered to shape: {movie_user_matrix_lite.shape}")

# Step 7: Create CSR matrix for KNN
print("\n7. Creating CSR matrix for KNN...")
csr_matrix_lite = sparse.csr_matrix(movie_user_matrix_lite.values)
print(f" CSR matrix shape: {csr_matrix_lite.shape}")

# Step 8: Train KNN model
print("\n8. Training KNN model...")
knn_model_lite = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11)
knn_model_lite.fit(csr_matrix_lite)
print(f" KNN model trained")

# Step 9: Compute SVD similarity matrix
print("\n9. Computing SVD similarity matrix...")
print("  (This may take a few minutes...)")
# Use cosine similarity on the movie-user matrix
svd_similarity_lite = cosine_similarity(movie_user_matrix_lite.fillna(0))
print(f" SVD similarity shape: {svd_similarity_lite.shape}")

# Step 10: Save all lite models
print("\n10. Saving lite models...")
with open("models_lite/movies_tag.pkl", "wb") as f:
    pickle.dump(movies_tag_lite, f)
print(" Saved movies_tag.pkl")

with open("models_lite/org_dataset.pkl", "wb") as f:
    pickle.dump(org_dataset_lite, f)
print("Saved org_dataset.pkl")

with open("models_lite/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix_lite, f)
print("Saved tfidf_matrix.pkl")

with open("models_lite/movie_user_matrix.pkl", "wb") as f:
    pickle.dump(movie_user_matrix_lite, f)
print("Saved movie_user_matrix.pkl")

with open("models_lite/csr_out_matrix.pkl", "wb") as f:
    pickle.dump(csr_matrix_lite, f)
print("Saved csr_out_matrix.pkl")

with open("models_lite/knn_model.pkl", "wb") as f:
    pickle.dump(knn_model_lite, f)
print("Saved knn_model.pkl")

with open("models_lite/svd_similarity.pkl", "wb") as f:
    pickle.dump(svd_similarity_lite, f)
print("Saved svd_similarity.pkl")

# Step 11: Calculate size reduction
print("\n" + "="*70)
print("SIZE COMPARISON")
print("="*70)

def get_dir_size(directory):
    total = 0
    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        if os.path.isfile(filepath):
            total += os.path.getsize(filepath)
    return total

original_size = get_dir_size("models") / (1024**3)  # GB
lite_size = get_dir_size("models_lite") / (1024**3)  # GB
reduction = ((original_size - lite_size) / original_size) * 100

print(f"\nOriginal models: {original_size:.2f} GB")
print(f"Lite models: {lite_size:.2f} GB")
print(f"Reduction: {reduction:.1f}%")
print(f"Savings: {original_size - lite_size:.2f} GB")

print("\n" + "="*70)
print("models created in 'models_lite/' directory")
