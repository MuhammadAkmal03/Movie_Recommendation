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


