"""
Script to replace movie_user_matrix with csr_out_matrix in recommender.py
"""
import re

# Read the file
with open('src/models/recommender.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Show current usage
print("=== Current movie_user_matrix usage ===")
lines = content.split('\n')
for i, line in enumerate(lines, 1):
    if 'movie_user_matrix' in line and 'models[' in line:
        print(f"Line {i}: {line.strip()}")

print("\n=== Analysis ===")
print("movie_user_matrix is used in:")
print("1. recommend_knn() - for KNN collaborative filtering")
print("2. recommend_svd() - for SVD collaborative filtering")
print("\nBoth use it to:")
print("- Get movie index: movie_user_matrix.index.get_loc(movie_id)")
print("- Access movie IDs: movie_user_matrix.index[idx]")
print("\nThe CSR matrix has the same structure, so we can:")
print("1. Convert CSR to DataFrame when needed")
print("2. Or keep a separate index mapping")
