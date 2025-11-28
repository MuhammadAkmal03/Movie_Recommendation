"""
A/B Testing Analytics Module

This module analyzes user feedback to compare different recommendation methods.
It helps answer: "Which recommendation method do users like most?"
"""
import json
import os
from collections import defaultdict

FEEDBACK_FILE = "feedback_data.json"

def _load_feedback():
    """Load feedback data"""
    if not os.path.exists(FEEDBACK_FILE):
        return {"ratings": []}
    with open(FEEDBACK_FILE, 'r') as f:
        return json.load(f)

def get_method_performance():
    """
    Analyze performance of each recommendation method
    
    Returns dict with metrics for each method:
    - total_ratings: How many times this method was rated
    - likes: Number of likes
    - dislikes: Number of dislikes
    - like_rate: Percentage of likes (0-100)
    """
    data = _load_feedback()
    
    # Group ratings by method
    method_stats = defaultdict(lambda: {"likes": 0, "dislikes": 0, "total": 0})
    
    for rating in data.get("ratings", []):
        method = rating.get("method", "Unknown")
        rating_type = rating.get("rating", "")
        
        if method and rating_type:
            method_stats[method]["total"] += 1
            if rating_type == "like":
                method_stats[method]["likes"] += 1
            else:
                method_stats[method]["dislikes"] += 1
    
    # Calculate like rates
    results = {}
    for method, stats in method_stats.items():
        total = stats["total"]
        likes = stats["likes"]
        like_rate = (likes / total * 100) if total > 0 else 0
        
        results[method] = {
            "total_ratings": total,
            "likes": likes,
            "dislikes": stats["dislikes"],
            "like_rate": round(like_rate, 1)
        }
    
    return results

def get_winning_method():
    """
    Determine which method has the highest like rate
    
    Returns tuple: (method_name, like_rate)
    """
    performance = get_method_performance()
    
    if not performance:
        return None, 0
    
    # Find method with highest like rate
    winner = max(performance.items(), key=lambda x: x[1]["like_rate"])
    return winner[0], winner[1]["like_rate"]

def get_ab_test_summary():
    """
    Get complete A/B test summary
    
    Returns dict with:
    - method_performance: Stats for each method
    - winner: Best performing method
    - total_tests: Total number of ratings
    """
    performance = get_method_performance()
    winner_method, winner_rate = get_winning_method()
    
    total_tests = sum(stats["total_ratings"] for stats in performance.values())
    
    return {
        "method_performance": performance,
        "winner": {
            "method": winner_method,
            "like_rate": winner_rate
        },
        "total_tests": total_tests
    }
