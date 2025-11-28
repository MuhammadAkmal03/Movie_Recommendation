"""
User feedback and ratings management
"""
import json
import os
from datetime import datetime
from pathlib import Path

FEEDBACK_FILE = "feedback_data.json"

def _load_feedback():
    """Load feedback data from JSON file"""
    if not os.path.exists(FEEDBACK_FILE):
        return {"ratings": [], "stats": {"total_ratings": 0, "likes": 0, "dislikes": 0}}
    
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"ratings": [], "stats": {"total_ratings": 0, "likes": 0, "dislikes": 0}}

def _save_feedback(data):
    """Save feedback data to JSON file"""
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def save_rating(movie_title, rating, query="", method=""):
    """
    Save a user rating
    
    Args:
        movie_title: Title of the rated movie
        rating: 'like' or 'dislike'
        query: Search query or source movie (optional)
        method: Recommendation method used (optional)
    """
    data = _load_feedback()
    
    # Add new rating
    rating_entry = {
        "timestamp": datetime.now().isoformat(),
        "movie": movie_title,
        "rating": rating,
        "query": query,
        "method": method
    }
    data["ratings"].append(rating_entry)
    
    # Update stats
    data["stats"]["total_ratings"] += 1
    if rating == "like":
        data["stats"]["likes"] += 1
    else:
        data["stats"]["dislikes"] += 1
    
    _save_feedback(data)
    
    # Log to MLflow
    try:
        from src.utils.mlflow_logger import log_user_rating
        log_user_rating(movie_title, rating, method)
    except Exception as e:
        # Silently fail if MLflow not available
        pass
    
    return True

def get_feedback_stats():
    """Get overall feedback statistics"""
    data = _load_feedback()
    stats = data["stats"]
    
    # Calculate percentages
    total = stats["total_ratings"]
    if total > 0:
        stats["like_percentage"] = round((stats["likes"] / total) * 100, 1)
        stats["dislike_percentage"] = round((stats["dislikes"] / total) * 100, 1)
    else:
        stats["like_percentage"] = 0
        stats["dislike_percentage"] = 0
    
    return stats

def get_liked_movies():
    """Get list of movies the user liked"""
    data = _load_feedback()
    return [r["movie"] for r in data["ratings"] if r["rating"] == "like"]

def get_disliked_movies():
    """Get list of movies the user disliked"""
    data = _load_feedback()
    return [r["movie"] for r in data["ratings"] if r["rating"] == "dislike"]

def get_user_preferences():
    """
    Analyze user preferences from feedback
    Returns dict with liked/disliked movies and patterns
    """
    data = _load_feedback()
    
    liked = [r["movie"] for r in data["ratings"] if r["rating"] == "like"]
    disliked = [r["movie"] for r in data["ratings"] if r["rating"] == "dislike"]
    
    return {
        "liked_movies": liked,
        "disliked_movies": disliked,
        "total_feedback": len(data["ratings"])
    }

def clear_feedback():
    """Clear all feedback data (for testing)"""
    data = {"ratings": [], "stats": {"total_ratings": 0, "likes": 0, "dislikes": 0}}
    _save_feedback(data)
    return True
