"""
Data loading module with caching
"""
import pickle
import streamlit as st
import sys
import os
from src import config

@st.cache_resource
def load_models():
    """
    Load all required models with error handling and caching
    """
    models = {}
    
    try:
        # Load required models
        with open(config.MOVIES_TAG_PATH, "rb") as f:
            models['movies_tag'] = pickle.load(f)
            
        with open(config.TFIDF_MATRIX_PATH, "rb") as f:
            models['tfidf_matrix'] = pickle.load(f)
            
        with open(config.ORG_DATASET_PATH, "rb") as f:
            models['org_dataset'] = pickle.load(f)
            
        # Load collaborative filtering models (optional for basic functionality)
        try:
            with open(config.MOVIE_USER_MATRIX_PATH, "rb") as f:
                models['movie_user_matrix'] = pickle.load(f)
            with open(config.CSR_MATRIX_PATH, "rb") as f:
                models['csr_matrix'] = pickle.load(f)
            with open(config.KNN_MODEL_PATH, "rb") as f:
                models['knn_model'] = pickle.load(f)
            with open(config.SVD_SIMILARITY_PATH, "rb") as f:
                models['svd_similarity'] = pickle.load(f)
        except FileNotFoundError:
            print("Warning: Collaborative filtering models not found. Some features will be disabled.")
            
        return models
        
    except FileNotFoundError as e:
        st.error(f"Critical Error: Required model file not found - {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
