# Movie Recommendation System

A production-ready Python project for building and serving movie recommendations using either:

- `timdb-bollywood` for Bollywood-focused recommendations
- `ml-latest-small` or `ml-25m` for MovieLens-based experimentation

The project includes:

- Popularity-based recommendations
- Content-based recommendations using TF-IDF and cosine similarity
- Collaborative filtering with user-based, item-based, and matrix factorization models
- A hybrid recommender that combines content similarity with collaborative relevance
- Evaluation with RMSE, MAE, Precision@K, and Recall@K
- EDA plots and a Streamlit interface

## Project Structure

```text
.
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── README.md
└── src/
    └── movie_recommender/
        ├── __init__.py
        ├── config.py
        ├── data.py
        ├── eda.py
        ├── evaluation.py
        ├── models.py
        ├── pipeline.py
        └── utils.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the System

The training script downloads the requested dataset automatically if it is missing.

```bash
python train.py --dataset timdb-bollywood --top-k 10
```

Optional flags:

```bash
python train.py --dataset timdb-bollywood --run-eda --tune-svd --top-k 10
```

Artifacts are saved to:

- `data/processed/`
- `models/`

## Launch the App

```bash
streamlit run app.py
```

The app automatically detects saved dataset bundles and defaults to the Bollywood TIMDB bundle when available.

## Public Deployment

This repository includes a lightweight public-serving app under `deploy/streamlit_public/` that is already configured for the Bollywood dataset.

- Streamlit Community Cloud entrypoint: `deploy/streamlit_public/app.py`
- Streamlit dependency file: `deploy/streamlit_public/requirements.txt`
- Render blueprint file: `render.yaml`
- Public artifact bundled in-repo: `deploy/streamlit_public/bollywood_public_bundle.joblib`

That public app serves precomputed Bollywood recommendations, so deployment does not need to retrain models or redownload datasets.

## What Each Recommender Does

### 1. Popularity-Based Filtering

Ranks movies using a weighted score derived from average rating and rating volume. This is useful for cold-start scenarios and for surfacing broadly liked movies.

### 2. Content-Based Filtering

Uses movie genres and user-generated tags to build a text profile for each movie. TF-IDF converts that metadata into vectors, and cosine similarity finds movies with similar content.

### 3. Collaborative Filtering

Learns from user-item interactions:

- User-based KNN: finds similar users and recommends what they liked
- Item-based KNN: finds similar movies based on user co-preference
- SVD: factorizes the interaction matrix into latent user and movie embeddings

### 4. Hybrid Model

Combines collaborative relevance from SVD with content similarity so the final ranking benefits from both user behavior and movie metadata.

## Trade-offs

- Popularity: simple, fast, but not personalized
- Content-based: interpretable and helpful for new items, but can overspecialize
- KNN collaborative: intuitive, but can be sparse and slower at scale
- SVD: usually stronger personalization, but less interpretable
- Hybrid: balanced and robust, but needs more components to maintain

## When to Use Which Model

- Use popularity for anonymous users or system bootstrapping
- Use content-based when metadata is rich and user history is limited
- Use collaborative filtering when you have enough interactions
- Use hybrid when you want stronger personalization with better cold-start behavior

## Reproducibility

- Random seeds are fixed in config
- Dataset download is automated
- Models and artifacts are persisted with versioned filenames

## Suggested Next Steps

- Swap to `ml-25m` for larger-scale experiments
- Add FastAPI for production serving
- Deploy the Streamlit app to Streamlit Community Cloud or another hosting platform
