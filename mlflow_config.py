"""
MLflow configuration
"""
import mlflow
import os

# Set tracking URI (local directory)
MLFLOW_TRACKING_URI = "file:///./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Experiment names
EXPERIMENT_MODEL_TRAINING = "model_training"
EXPERIMENT_RECOMMENDATIONS = "recommendation_performance"
EXPERIMENT_AB_TESTING = "ab_testing"

def setup_experiments():
    """Create MLflow experiments if they don't exist"""
    experiments = [
        EXPERIMENT_MODEL_TRAINING,
        EXPERIMENT_RECOMMENDATIONS,
        EXPERIMENT_AB_TESTING
    ]
    
    for exp_name in experiments:
        try:
            mlflow.create_experiment(exp_name)
            print(f"Created experiment: {exp_name}")
        except:
            print(f"Experiment already exists: {exp_name}")
    
    return True
