---
title: Movie Recommendation Engine
emoji: 🎬
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🎬 Movie Recommendation Engine

A comprehensive movie recommendation system implementing:
- **Content-Based Filtering**
- **Collaborative Filtering (KNN)**
- **Collaborative Filtering (SVD)**
- **Hybrid Recommendation** (Weighted Ensemble)

## 🚀 Features
- Interactive Streamlit UI
- Movie poster display (via TMDB)
- Multiple recommendation algorithms
- Hybrid engine for better accuracy

## 🛠️ Setup
This project requires large model files (>6GB). They are tracked using Git LFS.

## 📦 Models
- `movie_user_matrix.pkl` (4.9 GB)
- `svd_similarity.pkl` (759 MB)
- `csr_out_matrix.pkl` (225 MB)
- `knn_model.pkl` (225 MB)
