"""
Complete fix for Hugging Face deployment:
1. Update create_lite_models.py to save movie_index
2. Update loader.py to load movie_index instead of full matrix
3. Update recommender.py to use CSR matrix + index
"""
import pickle
import pandas as pd

print("Step 1: Creating movie_index.pkl from existing movie_user_matrix.pkl...")
print("(This is just the index - very small file)")

# Load the dense matrix just to extract the index
with open('models_lite/movie_user_matrix.pkl', 'rb') as f:
    movie_user_matrix = pickle.load(f)

# Save just the index (list of movie IDs)
movie_index = movie_user_matrix.index

with open('models_lite/movie_index.pkl', 'wb') as f:
    pickle.dump(movie_index, f)

print(f"✅ Created movie_index.pkl ({len(movie_index)} movies)")
print(f"   Size: ~{len(movie_index) * 8 / 1024:.1f} KB (vs 2.5GB for full matrix)")

print("\nStep 2: You can now DELETE movie_user_matrix.pkl")
print("The app will use:")
print("  - csr_out_matrix.pkl (172MB) - the actual data")
print("  - movie_index.pkl (~few KB) - just the movie IDs")
