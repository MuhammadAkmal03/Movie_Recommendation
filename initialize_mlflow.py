"""
Initialize MLflow experiments
"""
from mlflow_config import setup_experiments
import mlflow

print(" Initializing MLflow...")
print("=" * 50)

# Setup experiments
setup_experiments()

print("\n MLflow initialized successfully!")
print("\n Experiments created:")
print(" model_training")
print("recommendation_performance")
print(" ab_testing")

