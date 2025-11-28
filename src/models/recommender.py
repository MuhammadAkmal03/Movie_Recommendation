"""
Recommendation logic module
"""
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.helpers import normalize_title, get_poster_by_id
from src import config
import pandas as pd

def search_by_keywords(query, models, n_results=10):
    """
    NLP-powered keyword search using TF-IDF similarity
    
    Args:
        query: User's search query (e.g., "Tom Cruise action movies")
        models: Dictionary containing loaded models
        n_results: Number of results to return
        
    Returns:
        List of tuples (title, poster_url, score) sorted by relevance
    """
    movies_tag = models['movies_tag']
    org_dataset = models['org_dataset']
    
    if not query or query.strip() == "":
        return []
    
    # Normalize query
    query = query.lower().strip()
    
    # Simple keyword matching approach
    # Split query into keywords
    keywords = query.split()
    
    # Calculate match scores for each movie
    scores = []
    for idx, row in movies_tag.iterrows():
        tag = str(row['tag']).lower()
        
        # Count how many keywords appear in the tag
        matches = sum(1 for keyword in keywords if keyword in tag)
        
        # Calculate score based on matches and tag length
        if matches > 0:
            # Higher score for more matches, normalized by query length
            score = matches / len(keywords)
            # Boost score if all keywords match
            if matches == len(keywords):
                score *= 1.5
            scores.append((idx, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N results
    top_results = scores[:n_results]
    
    if len(top_results) == 0:
        return []
    
    # Get movie details
    result = []
    for idx, score in top_results:
        title = movies_tag.iloc[idx]['title']
        movie_id = movies_tag.iloc[idx]['id']
        poster_url = get_poster_by_id(movie_id, org_dataset)
        result.append((title, poster_url, score))
    
    return result

def recommend_content(movie_title, models):
    """
    Content-based recommendation with explanations
    """
    movies_tag = models['movies_tag']
    tfidf_matrix = models['tfidf_matrix']
    org_dataset = models['org_dataset']
    
    movie_title = normalize_title(movie_title)
    if movie_title not in movies_tag['title'].values:
        return []

    idx = movies_tag[movies_tag['title'] == movie_title].index[0]
    source_movie_id = movies_tag.iloc[idx]['id']
    
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = sim_scores.argsort()[-6:-1][::-1]

    result = []
    for i in similar_indices:
        title = movies_tag.iloc[i]['title']
        movie_id = movies_tag.iloc[i]['id']
        poster_url = get_poster_by_id(movie_id, org_dataset)
        similarity_score = sim_scores[i]
        
        # Generate explanation
        from src.utils.explainability import explain_recommendation
        explanations = explain_recommendation(source_movie_id, movie_id, org_dataset, similarity_score)
        
        result.append((title, poster_url, explanations))
    return result

def recommend_knn(movie_title, models):
    """
    Collaborative filtering (KNN)
    """
    if 'knn_model' not in models:
        return []
        
    org_dataset = models['org_dataset']
    # Use CSR matrix instead of dense matrix
    csr_matrix = models['csr_out_matrix']
    movie_index = models['movie_index']  # Just the index, very small
    knn_model = models['knn_model']
    
    movie_title = normalize_title(movie_title)
    match = org_dataset[org_dataset['title'] == movie_title]
    if match.empty:
        return []

    movie_id = match.iloc[0]['id']
    if movie_id not in movie_index:
        # Movie exists but wasn't in the collaborative filtering training set
        # Return empty with a note (will be handled in UI)
        return []

    movie_idx = movie_index.get_loc(movie_id)
    
    # Ensure n_neighbors doesn't exceed dataset size
    n_neighbors = min(11, len(movie_index))
    distances, indices = knn_model.kneighbors(
        csr_matrix[movie_idx], 
        n_neighbors=n_neighbors
    )

    result = []
    for i in range(1, len(indices[0])):
        sim_id = movie_index[indices[0][i]]
        row = org_dataset[org_dataset['id'] == sim_id]
        if not row.empty:
            title = row.iloc[0]['title']
            poster_url = get_poster_by_id(sim_id, org_dataset)
            
            # Generate explanation
            from src.utils.explainability import explain_recommendation
            explanations = explain_recommendation(movie_id, sim_id, org_dataset)
            explanations.insert(0, "Collaborative filtering: Users with similar taste liked this")
            
            result.append((title, poster_url, explanations))
    return result[:5]

def recommend_svd(movie_title, models):
    """
    Collaborative filtering (SVD)
    """
    if 'svd_similarity' not in models:
        return []
        
    org_dataset = models['org_dataset']
    # Use CSR matrix instead of dense matrix
    csr_matrix = models['csr_out_matrix']
    movie_index = models['movie_index']  # Just the index, very small
    svd_similarity = models['svd_similarity']
    
    movie_title = normalize_title(movie_title)
    match = org_dataset[org_dataset['title'] == movie_title]
    if match.empty:
        return []

    movie_id = match.iloc[0]['id']
    if movie_id not in movie_index:
        return []

    movie_idx = movie_index.get_loc(movie_id)
    scores = list(enumerate(svd_similarity[movie_idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]

    result = []
    for idx, score in scores:
        sim_id = movie_index[idx]
        row = org_dataset[org_dataset['id'] == sim_id]
        if not row.empty:
            title = row.iloc[0]['title']
            poster_url = get_poster_by_id(sim_id, org_dataset)
            
            # Generate explanation
            from src.utils.explainability import explain_recommendation
            explanations = explain_recommendation(movie_id, sim_id, org_dataset, score)
            explanations.insert(0, "SVD analysis: Highly correlated with your preferences")
            
            result.append((title, poster_url, explanations))
    return result[:5]

def recommend_hybrid(movie_title, models, weights=None, n_recommendations=5):
    """
    Hybrid recommendation with explanations
    """
    if weights is None:
        weights = config.HYBRID_WEIGHTS
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    content_recs = recommend_content(movie_title, models)
    knn_recs = recommend_knn(movie_title, models)
    svd_recs = recommend_svd(movie_title, models)
    
    if not content_recs and not knn_recs and not svd_recs:
        return []
    
    # Combine results with explanations
    all_recommendations = {}
    for title, poster_url, explanations in content_recs:
        all_recommendations[title] = (poster_url, explanations)
    
    for title, poster_url, explanations in knn_recs:
        if title not in all_recommendations:
            all_recommendations[title] = (poster_url, explanations)
        else:
            # Merge explanations if movie appears in both
            existing_poster, existing_exp = all_recommendations[title]
            merged_exp = list(set(existing_exp + explanations))  # Remove duplicates
            all_recommendations[title] = (existing_poster, merged_exp)
    
    for title, poster_url, explanations in svd_recs:
        if title not in all_recommendations:
            all_recommendations[title] = (poster_url, explanations)
        else:
            existing_poster, existing_exp = all_recommendations[title]
            merged_exp = list(set(existing_exp + explanations))
            all_recommendations[title] = (existing_poster, merged_exp)
    
    # Return top 5 with format: (title, poster_url, explanations)
    result = [(title, data[0], data[1]) for title, data in list(all_recommendations.items())[:n_recommendations]]
    return result

def get_recommendations(movie_name, method, models):
    """
    Main entry point for recommendations
    """
    import time
    start_time = time.time()
    
    # Get recommendations
    if method == "Content Based Filtering":
        results = recommend_content(movie_name, models)
    elif method == "Collaborative Filtering (KNN)":
        results = recommend_knn(movie_name, models)
    elif method == "Collaborative Filtering (SVD)":
        results = recommend_svd(movie_name, models)
    elif method == "Hybrid Recommendation":
        results = recommend_hybrid(movie_name, models)
    else:
        results = []
    
    # Calculate metrics and log to MLflow
    response_time_ms = (time.time() - start_time) * 1000
    num_results = len(results)
    
    try:
        from src.utils.mlflow_logger import log_recommendation_performance
        log_recommendation_performance(
            method=method,
            query=movie_name,
            num_results=num_results,
            response_time_ms=response_time_ms
        )
    except Exception as e:
        # Silently fail if MLflow not available
        pass
    
    return results
