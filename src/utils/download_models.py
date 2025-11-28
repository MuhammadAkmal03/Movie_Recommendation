import os
import requests
import streamlit as st
from pathlib import Path

def download_file(url, dest_path):
    """Download a file from URL to destination path with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    with open(dest_path, 'wb') as f:
        with st.spinner(f"Downloading {os.path.basename(dest_path)}..."):
            progress_bar = st.progress(0)
            downloaded = 0
            for data in response.iter_content(block_size):
                downloaded += len(data)
                f.write(data)
                if total_size > 0:
                    progress_bar.progress(min(downloaded / total_size, 1.0))
            progress_bar.empty()

def check_and_download_models():
    """Check if models exist, download from GitHub Releases if missing."""
    MODELS_DIR = "models_lite"
    
    # Create directory if it doesn't exist
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    # List of files to check/download
    # UPDATE THIS URL AFTER CREATING THE RELEASE
    RELEASE_URL = "https://github.com/MuhammadAkmal03/Movie_Recommendation/releases/download/v1.0-models"
    
    files = [
        "csr_out_matrix.pkl",
        "knn_model.pkl",
        "movie_index.pkl",
        "movies_tag.pkl",
        "org_dataset.pkl",
        "svd_similarity.pkl",
        "tfidf_matrix.pkl"
    ]
    
    missing_files = []
    for file in files:
        file_path = os.path.join(MODELS_DIR, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
            
    if missing_files:
        st.info(f"Downloading {len(missing_files)} model files... This only happens once.")
        for file in missing_files:
            url = f"{RELEASE_URL}/{file}"
            dest_path = os.path.join(MODELS_DIR, file)
            try:
                download_file(url, dest_path)
            except Exception as e:
                st.error(f"Failed to download {file}: {str(e)}")
                st.stop()
        st.success("All models downloaded successfully!")
