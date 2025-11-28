"""
Helper functions for the Movie Recommendation System
"""
import re
import pandas as pd
from src.config import TMDB_IMAGE_BASE_URL

def normalize_title(title):
    """
    Normalize movie title for consistent searching
    """
    if isinstance(title, str):
        title = title.lower()
        title = re.sub(r'[^a-z0-9\s]', '', title)
        title = re.sub(r'\s+', ' ', title).strip()
    return title

def get_poster_by_id(movie_id, org_dataset):
    """
    Get poster URL for a movie ID
    """
    # Convert to int to handle type mismatch between datasets
    try:
        movie_id = int(movie_id)
    except (ValueError, TypeError):
        return ''
    
    row = org_dataset[org_dataset['id'] == movie_id]
    if not row.empty:
        return row.iloc[0].get('poster_url', '')
    return ''

def ensure_poster_url(org_dataset):
    """
    Ensure poster_url column exists in dataset
    """
    if 'poster_url' not in org_dataset.columns and 'poster_path' in org_dataset.columns:
        org_dataset['poster_url'] = org_dataset['poster_path'].apply(
            lambda x: TMDB_IMAGE_BASE_URL + x if pd.notnull(x) else ""
        )
    return org_dataset
