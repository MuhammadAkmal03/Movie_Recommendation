"""
Update loader.py to load movie_index instead of movie_user_matrix
"""

# Read the loader file
with open('src/data/loader.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the loading of movie_user_matrix with movie_index
content = content.replace(
    "models['movie_user_matrix'] = pickle.load(f)",
    "models['movie_index'] = pickle.load(f)  # Just the index, not the full matrix"
)

content = content.replace(
    "MOVIE_USER_MATRIX_PATH",
    "MOVIE_INDEX_PATH"
)

# Write back
with open('src/data/loader.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(" Updated src/data/loader.py")
print("   Changed: MOVIE_USER_MATRIX_PATH -> MOVIE_INDEX_PATH")

