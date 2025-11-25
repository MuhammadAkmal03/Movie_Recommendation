
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load precomputed objects
movies_tag = pickle.load(open("models/movies_tag.pkl", "rb"))
tfidf_matrix = pickle.load(open("models/tfidf_matrix.pkl", "rb"))
movie_user_matrix = pickle.load(open("models/movie_user_matrix.pkl", "rb"))
csr_out_matrix = pickle.load(open("models/csr_out_matrix.pkl", "rb"))
knn_model = pickle.load(open("models/knn_model.pkl", "rb"))
svd_similarity = pickle.load(open("models/svd_similarity.pkl", "rb"))
org_dataset = pickle.load(open("models/org_dataset.pkl", "rb"))

# Create poster_url column if not present
if 'poster_url' not in org_dataset.columns and 'poster_path' in org_dataset.columns:
    base_url = "https://image.tmdb.org/t/p/w500"
    org_dataset['poster_url'] = org_dataset['poster_path'].apply(lambda x: base_url + x if pd.notnull(x) else "")

def normalize_title(title):
    if isinstance(title, str):
        title = title.lower()
        title = re.sub(r'[^a-z0-9\s]', '', title)
        title = re.sub(r'\s+', ' ', title).strip()
    return title

def get_poster_by_id(movie_id):
    # Convert to int to handle type mismatch between datasets
    try:
        original_id = movie_id
        movie_id = int(movie_id)
        print(f"[DEBUG] get_poster_by_id: {original_id} ({type(original_id)}) -> {movie_id} (int)")
    except (ValueError, TypeError):
        print(f"[DEBUG] get_poster_by_id: Failed to convert {movie_id}")
        return ''
    
    row = org_dataset[org_dataset['id'] == movie_id]
    print(f"[DEBUG] Found {len(row)} rows for movie_id {movie_id}")
    
    if not row.empty:
        poster_url = row.iloc[0].get('poster_url', '')
        print(f"[DEBUG] Poster URL: {poster_url[:60] if poster_url else 'EMPTY'}")
        return poster_url
    return ''

def recommend_content(movie_title):
    movie_title = normalize_title(movie_title)
    if movie_title not in movies_tag['title'].values:
        return []

    idx = movies_tag[movies_tag['title'] == movie_title].index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = sim_scores.argsort()[-6:-1][::-1]

    result = []
    for i in similar_indices:
        title = movies_tag.iloc[i]['title']
        movie_id = movies_tag.iloc[i]['id']
        poster_url = get_poster_by_id(movie_id)
        result.append((title, poster_url))
    return result

def recommend_knn(movie_title):
    movie_title = normalize_title(movie_title)
    match = org_dataset[org_dataset['title'] == movie_title]
    if match.empty:
        return []

    movie_id = match.iloc[0]['id']
    if movie_id not in movie_user_matrix.index:
        return []

    movie_idx = movie_user_matrix.index.get_loc(movie_id)
    distances, indices = knn_model.kneighbors(csr_out_matrix[movie_idx], n_neighbors=11)

    result = []
    for i in range(1, len(indices[0])):
        sim_id = movie_user_matrix.index[indices[0][i]]
        row = org_dataset[org_dataset['id'] == sim_id]
        if not row.empty:
            title = row.iloc[0]['title']
            poster_url = row.iloc[0].get('poster_url', '')
            result.append((title, poster_url))
    return result

def recommend_svd(movie_title):
    movie_title = normalize_title(movie_title)
    match = org_dataset[org_dataset['title'] == movie_title]
    if match.empty:
        return []

    movie_id = match.iloc[0]['id']
    if movie_id not in movie_user_matrix.index:
        return []

    movie_idx = movie_user_matrix.index.get_loc(movie_id)
    scores = list(enumerate(svd_similarity[movie_idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]

    result = []
    for idx, score in scores:
        sim_id = movie_user_matrix.index[idx]
        row = org_dataset[org_dataset['id'] == sim_id]
        if not row.empty:
            title = row.iloc[0]['title']
            poster_url = row.iloc[0].get('poster_url', '')
            result.append((title, poster_url))
    return result

def recommend_hybrid(movie_name, weights=None, n_recommendations=5):
    """
    Hybrid recommendation combining Content-Based, KNN, and SVD methods.
    
    Args:
        movie_name: Movie title to get recommendations for
        weights: Dict with keys 'content', 'knn', 'svd' (default: {content: 0.3, knn: 0.3, svd: 0.4})
        n_recommendations: Number of recommendations to return
        
    Returns:
        List of tuples (title, poster_url) sorted by combined score
    """
    # Default weights
    if weights is None:
        weights = {'content': 0.3, 'knn': 0.3, 'svd': 0.4}
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Get recommendations from each method (get more than needed for diversity)
    n_fetch = n_recommendations * 3
    
    content_recs = recommend_content(movie_name)
    knn_recs = recommend_knn(movie_name)
    svd_recs = recommend_svd(movie_name)
    
    # If all methods fail, return empty
    if not content_recs and not knn_recs and not svd_recs:
        return []
    
    # Score each movie based on its position in each method's results
    # Higher position = higher score
    movie_scores = {}
    
    # Process content-based recommendations
    for idx, (title, poster_url) in enumerate(content_recs):
        # Score decreases with position: 1st place = highest score
        score = (len(content_recs) - idx) / len(content_recs)
        if title not in movie_scores:
            movie_scores[title] = {'score': 0, 'poster_url': poster_url, 'methods': []}
        movie_scores[title]['score'] += score * weights['content']
        movie_scores[title]['methods'].append('Content')
    
    # Process KNN recommendations
    for idx, (title, poster_url) in enumerate(knn_recs):
        score = (len(knn_recs) - idx) / len(knn_recs)
        if title not in movie_scores:
            movie_scores[title] = {'score': 0, 'poster_url': poster_url, 'methods': []}
        movie_scores[title]['score'] += score * weights['knn']
        movie_scores[title]['methods'].append('KNN')
        # Update poster_url if not set
        if not movie_scores[title]['poster_url']:
            movie_scores[title]['poster_url'] = poster_url
    
    # Process SVD recommendations
    for idx, (title, poster_url) in enumerate(svd_recs):
        score = (len(svd_recs) - idx) / len(svd_recs)
        if title not in movie_scores:
            movie_scores[title] = {'score': 0, 'poster_url': poster_url, 'methods': []}
        movie_scores[title]['score'] += score * weights['svd']
        movie_scores[title]['methods'].append('SVD')
        # Update poster_url if not set
        if not movie_scores[title]['poster_url']:
            movie_scores[title]['poster_url'] = poster_url
    
    # Sort by combined score (descending)
    sorted_movies = sorted(
        movie_scores.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    # Return top N recommendations
    result = [
        (title, data['poster_url'])
        for title, data in sorted_movies[:n_recommendations]
    ]
    
    return result

def recommend_movies(movie_name, method):
    if method == "Content Based Filtering":
        return recommend_content(movie_name)
    elif method == "Collaborative Filtering (KNN)":
        return recommend_knn(movie_name)
    elif method == "Collaborative Filtering (SVD)":
        return recommend_svd(movie_name)
    elif method == "Hybrid Recommendation":
        return recommend_hybrid(movie_name)
    else:
        return []
