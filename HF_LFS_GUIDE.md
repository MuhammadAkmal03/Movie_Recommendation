# Hugging Face Deployment - Git LFS Solution

## Problem
Your `models_lite/` folder (2.5GB) exceeds Hugging Face's 1GB storage limit.

## Solution: Use Git LFS

### Step 1: Verify Git LFS is Set Up
```bash
# Already done ✅
git lfs install
git lfs track "*.pkl"
```

### Step 2: Remove models_lite from .gitignore
Edit `.gitignore` and **comment out** or **remove** this line:
```
# models_lite/   <-- Comment this out
```

### Step 3: Add and Commit with LFS
```bash
# Add .gitattributes
git add .gitattributes

# Add model files (will use LFS automatically)
git add models_lite/

# Add other files
git add .streamlit/ requirements.txt src/ app.py assets/ README.md

# Commit
git commit -m "feat: Add models with Git LFS for HF deployment"
```

### Step 4: Push to Hugging Face
```bash
# Add Hugging Face remote (replace YOUR_USERNAME)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/movie-recommendation-ai

# Push (this will upload large files via LFS)
git push hf main
```

**Note:** The push will take 10-30 minutes due to large files.

---

## Alternative: Reduce Model Size

If Git LFS doesn't work, you can reduce the model size:

### Option A: Remove the largest file
The `movie_user_matrix.pkl` (2.5GB) is the biggest. If you're not using it:
```bash
# Delete it
rm models_lite/movie_user_matrix.pkl

# Update your code to not load it
# (You already have csr_out_matrix.pkl which is smaller)
```

### Option B: Compress models
Use `joblib` with compression:
```python
import joblib
joblib.dump(model, 'model.pkl', compress=3)
```

---

## Quick Commands

```bash
# 1. Update .gitignore (remove models_lite/)
# 2. Add everything
git add .

# 3. Commit
git commit -m "feat: Deploy to Hugging Face with Git LFS"

# 4. Push to HF
git push hf main
```

---

## Verify LFS is Working

Check if `.pkl` files are tracked:
```bash
git lfs ls-files
```

You should see your model files listed.
