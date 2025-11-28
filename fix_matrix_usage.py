"""
Fix script: Replace movie_user_matrix with csr_out_matrix + index
This reduces size from 2.5GB to ~172MB
"""
import re

# Read the file
with open('src/models/recommender.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace movie_user_matrix references
# The key insight: we only need the INDEX, not the full dense matrix
# The CSR matrix already has the data

replacements = [
    # In recommend_knn function
    (
        "movie_user_matrix = models['movie_user_matrix']",
        "# Use CSR matrix instead of dense matrix\n    csr_matrix = models['csr_out_matrix']\n    movie_index = models['movie_index']  # Just the index, very small"
    ),
    (
        "if movie_id not in movie_user_matrix.index:",
        "if movie_id not in movie_index:"
    ),
    (
        "movie_idx = movie_user_matrix.index.get_loc(movie_id)",
        "movie_idx = movie_index.get_loc(movie_id)"
    ),
    (
        "n_neighbors = min(11, len(movie_user_matrix))",
        "n_neighbors = min(11, len(movie_index))"
    ),
    (
        "sim_id = movie_user_matrix.index[indices[0][i]]",
        "sim_id = movie_index[indices[0][i]]"
    ),
    (
        "sim_id = movie_user_matrix.index[idx]",
        "sim_id = movie_index[idx]"
    ),
]

# Apply replacements
for old, new in replacements:
    content = content.replace(old, new)

# Write back
with open('src/models/recommender.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Updated src/models/recommender.py")
print("\nChanges made:")
print("- Replaced movie_user_matrix with csr_out_matrix + movie_index")
print("- movie_index is just the list of movie IDs (very small, ~few KB)")
print("- CSR matrix contains the actual data (172MB instead of 2.5GB)")
