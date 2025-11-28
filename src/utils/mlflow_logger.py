"""
MLflow logging utilities
"""
import mlflow
import time
from datetime import datetime

def log_recommendation_performance(method, query, num_results, response_time_ms, avg_similarity=None):
    """
    Log recommendation performance
    
    Args:
        method: Recommendation method used
        query: Search query or movie name
        num_results: Number of results returned
        response_time_ms: Time taken in milliseconds
        avg_similarity: Average similarity score (optional)
    """
    try:
        mlflow.set_experiment("recommendation_performance")
        
        with mlflow.start_run(run_name=f"{method}_{datetime.now().strftime('%H%M%S')}"):
            # Log parameters
            mlflow.log_param("method", method)
            mlflow.log_param("query", query[:50] if query else "")  # Truncate long queries
            
            # Log metrics
            mlflow.log_metric("num_results", num_results)
            mlflow.log_metric("response_time_ms", response_time_ms)
            
            if avg_similarity:
                mlflow.log_metric("avg_similarity", avg_similarity)
            
            # Log tags
            mlflow.set_tag("timestamp", datetime.now().isoformat())
    except Exception as e:
        # Silently fail - don't break app if MLflow fails
        print(f"MLflow logging failed: {e}")

def log_ab_test_metrics(method_performance):
    """
    Log A/B testing results
    
    Args:
        method_performance: Dict from get_ab_test_summary()
    """
    try:
        mlflow.set_experiment("ab_testing")
        
        with mlflow.start_run(run_name=f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log overall metrics
            for method, stats in method_performance.items():
                # Clean method name for MLflow (replace spaces with underscores)
                clean_method = method.replace(" ", "_").replace("(", "").replace(")", "")
                
                mlflow.log_metric(f"{clean_method}_like_rate", stats["like_rate"])
                mlflow.log_metric(f"{clean_method}_total_ratings", stats["total_ratings"])
                mlflow.log_metric(f"{clean_method}_likes", stats["likes"])
                mlflow.log_metric(f"{clean_method}_dislikes", stats["dislikes"])
            
            # Log tags
            mlflow.set_tag("test_date", datetime.now().isoformat())
            
            print("✅ Logged A/B test results to MLflow")
            return True
    except Exception as e:
        print(f"MLflow logging failed: {e}")
        return False

def log_user_rating(movie, rating, method):
    """
    Log individual user rating
    
    Args:
        movie: Movie title
        rating: 'like' or 'dislike'
        method: Recommendation method
    """
    try:
        mlflow.set_experiment("ab_testing")
        
        with mlflow.start_run(run_name=f"rating_{datetime.now().strftime('%H%M%S')}"):
            mlflow.log_param("movie", movie[:50] if movie else "")
            mlflow.log_param("method", method)
            mlflow.log_metric("rating", 1 if rating == "like" else 0)
            mlflow.set_tag("timestamp", datetime.now().isoformat())
    except Exception as e:
        # Silently fail
        print(f"MLflow logging failed: {e}")
