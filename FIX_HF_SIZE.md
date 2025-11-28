# Solution: Remove movie_user_matrix.pkl for Hugging Face Deployment

## The Problem
- `movie_user_matrix.pkl` = **2.5GB** (exceeds HF's 1GB limit)
- You already have `csr_out_matrix.pkl` (172MB) which is the sparse version
- The dense matrix is redundant!

## Solution: Delete the Large File

### Step 1: Remove movie_user_matrix.pkl
```bash
# Delete the file
rm models_lite/movie_user_matrix.pkl

# Or on Windows
del models_lite\movie_user_matrix.pkl
```

### Step 2: Update Code (if needed)
Check if your code uses `movie_user_matrix.pkl`. If it does, replace it with `csr_out_matrix.pkl` (the sparse version).

In `src/data/loader.py`, ensure you're loading the CSR matrix instead.

### Step 3: Commit and Push
```bash
# Add the deletion
git add models_lite/

# Commit
git commit -m "Remove dense matrix, use sparse CSR matrix for HF deployment"

# Push to Hugging Face
git push hf main
```

## New Total Size (without movie_user_matrix.pkl)
- `csr_out_matrix.pkl`: 172 MB
- `knn_model.pkl`: 172 MB
- `svd_similarity.pkl`: 50 MB
- `org_dataset.pkl`: 29 MB
- `movies_tag.pkl`: 21 MB
- `tfidf_matrix.pkl`: 15 MB
- **Total: ~460 MB** ✅ (Under 1GB!)

## Verify Your Code Doesn't Need It
```bash
# Search for references to movie_user_matrix
grep -r "movie_user_matrix" src/
```

If nothing critical uses it, you're safe to delete it!
