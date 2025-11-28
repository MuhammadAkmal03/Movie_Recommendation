"""
Recommendation Explainability Module

Generates human-readable explanations for why movies are recommended
"""
import json
import ast

def parse_json_field(field_value):
    """
    Parse JSON-like fields (genres, keywords, cast, crew)
    Returns list of names
    """
    if not field_value or field_value == '[]':
        return []
    
    try:
        # Try parsing as JSON
        if isinstance(field_value, str):
            data = json.loads(field_value.replace("'", '"'))
        else:
            data = field_value
        
        # Extract names
        if isinstance(data, list):
            return [item.get('name', item) if isinstance(item, dict) else str(item) for item in data]
        return []
    except:
        try:
            # Try ast.literal_eval for Python-style strings
            data = ast.literal_eval(str(field_value))
            if isinstance(data, list):
                return [item.get('name', item) if isinstance(item, dict) else str(item) for item in data]
        except:
            return []

def get_movie_features(movie_row):
    """
    Extract features from a movie row
    
    Returns dict with:
    - genres: list of genre names
    - keywords: list of keywords
    - year: release year
    """
    features = {
        'genres': [],
        'keywords': [],
        'year': None
    }
    
    # Extract genres
    if 'genres' in movie_row.index:
        features['genres'] = parse_json_field(movie_row['genres'])
    
    # Extract keywords
    if 'keywords' in movie_row.index:
        features['keywords'] = parse_json_field(movie_row['keywords'])
    
    # Extract year
    if 'release_date' in movie_row.index and movie_row['release_date']:
        try:
            features['year'] = int(str(movie_row['release_date'])[:4])
        except:
            pass
    
    return features

def generate_explanation(source_features, recommended_features, similarity_score=None):
    """
    Generate explanation for why a movie was recommended
    
    Args:
        source_features: Features of the source movie
        recommended_features: Features of the recommended movie
        similarity_score: Optional similarity score (0-1)
    
    Returns:
        List of explanation strings
    """
    explanations = []
    
    # Compare genres
    if source_features['genres'] and recommended_features['genres']:
        common_genres = set(source_features['genres']) & set(recommended_features['genres'])
        if common_genres:
            genre_str = ', '.join(list(common_genres)[:2])  # Show max 2 genres
            explanations.append(f"Similar genres: {genre_str}")
    
    # Compare keywords/themes
    if source_features['keywords'] and recommended_features['keywords']:
        common_keywords = set(source_features['keywords']) & set(recommended_features['keywords'])
        if common_keywords:
            # Show top 2 common keywords
            keyword_str = ', '.join(list(common_keywords)[:2])
            explanations.append(f"Similar themes: {keyword_str}")
    
    # Compare year
    if source_features['year'] and recommended_features['year']:
        year_diff = abs(source_features['year'] - recommended_features['year'])
        if year_diff <= 3:
            explanations.append(f"Similar era: {recommended_features['year']}")
    
    # Add similarity score
    if similarity_score is not None:
        explanations.append(f"Match score: {similarity_score:.0%}")
    
    # If no specific reasons, add generic one
    if not explanations and similarity_score:
        explanations.append(f"Content similarity: {similarity_score:.0%}")
    
    return explanations

def explain_recommendation(source_movie_id, recommended_movie_id, org_dataset, similarity_score=None):
    """
    Generate explanation for a recommendation
    
    Args:
        source_movie_id: ID of the source movie
        recommended_movie_id: ID of the recommended movie
        org_dataset: DataFrame with movie data
        similarity_score: Optional similarity score
    
    Returns:
        List of explanation strings
    """
    # Get source movie
    source_row = org_dataset[org_dataset['id'] == source_movie_id]
    if source_row.empty:
        return ["Recommended based on content similarity"]
    
    # Get recommended movie
    rec_row = org_dataset[org_dataset['id'] == recommended_movie_id]
    if rec_row.empty:
        return ["Recommended based on content similarity"]
    
    # Extract features
    source_features = get_movie_features(source_row.iloc[0])
    rec_features = get_movie_features(rec_row.iloc[0])
    
    # Generate explanation
    return generate_explanation(source_features, rec_features, similarity_score)
