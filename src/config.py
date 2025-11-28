"""
Configuration settings for the Movie Recommendation System
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, "models_lite")
MOVIES_TAG_PATH = os.path.join(MODELS_DIR, "movies_tag.pkl")
TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
MOVIE_INDEX_PATH = os.path.join(MODELS_DIR, "movie_index.pkl")  # Just the index, not the full matrix
CSR_MATRIX_PATH = os.path.join(MODELS_DIR, "csr_out_matrix.pkl")
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")
SVD_SIMILARITY_PATH = os.path.join(MODELS_DIR, "svd_similarity.pkl")
ORG_DATASET_PATH = os.path.join(MODELS_DIR, "org_dataset.pkl")

# Recommendation settings
DEFAULT_RECOMMENDATIONS = 5
HYBRID_WEIGHTS = {'content': 0.3, 'knn': 0.3, 'svd': 0.4}

# TMDB settings
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
