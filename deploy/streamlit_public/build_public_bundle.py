"""Build a lightweight Bollywood bundle for public Streamlit deployment."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_PATH = Path(__file__).resolve().parent / "bollywood_public_bundle.joblib"
SOURCE_BUNDLE_PATH = MODELS_DIR / "timdb-bollywood_recommender_bundle.joblib"
TOP_N = 100


def _minmax(series: pd.Series) -> pd.Series:
    """Normalize values into the [0, 1] range."""
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_value) / (max_value - min_value)


def build_movies_table(movies: pd.DataFrame) -> pd.DataFrame:
    """Create the compact movies table used by the public app."""
    public_movies = movies.copy()
    public_movies["release_year"] = public_movies["release_year"].fillna("").astype(str)
    public_movies["display_title"] = public_movies["title"].astype(str)
    release_mask = public_movies["release_year"].str.strip().ne("") & public_movies["release_year"].ne("Unknown")
    public_movies.loc[release_mask, "display_title"] = (
        public_movies.loc[release_mask, "title"].astype(str)
        + " ("
        + public_movies.loc[release_mask, "release_year"].astype(str)
        + ")"
    )
    return public_movies[["movieId", "title", "display_title", "genres", "release_year"]].drop_duplicates("movieId")


def build_content_table(recommender, movies_table: pd.DataFrame) -> pd.DataFrame:
    """Precompute content recommendations per Bollywood title."""
    rows: list[pd.DataFrame] = []
    for movie_id in movies_table["movieId"].tolist():
        scores = recommender.similarity_scores_for_movie(int(movie_id)).copy()
        scores = scores.loc[scores["movieId"] != movie_id].copy()
        scores = scores.sort_values("content_score", ascending=False).head(TOP_N).reset_index(drop=True)
        scores["content_score"] = _minmax(scores["content_score"]).fillna(0.0)
        scores["rank"] = scores.index + 1
        scores["anchor_movieId"] = int(movie_id)
        rows.append(scores[["anchor_movieId", "movieId", "content_score", "rank"]])

    content_table = pd.concat(rows, ignore_index=True)
    return content_table.merge(
        movies_table[["movieId", "title", "display_title", "genres", "release_year"]],
        on="movieId",
        how="left",
    )


def build_collaborative_table(recommender, ratings: pd.DataFrame, movies_table: pd.DataFrame) -> pd.DataFrame:
    """Precompute collaborative SVD recommendations per user."""
    rows: list[pd.DataFrame] = []
    for user_id in sorted(ratings["userId"].unique().tolist()):
        user_recommendations = recommender.recommend_for_user(
            user_id=int(user_id),
            top_k=TOP_N,
            algorithm="svd",
        ).copy()
        user_recommendations["collaborative_score"] = _minmax(user_recommendations["predicted_rating"]).fillna(0.0)
        user_recommendations["rank"] = range(1, len(user_recommendations) + 1)
        user_recommendations["userId"] = int(user_id)
        rows.append(
            user_recommendations[
                ["userId", "movieId", "predicted_rating", "collaborative_score", "rank"]
            ]
        )

    collaborative_table = pd.concat(rows, ignore_index=True)
    return collaborative_table.merge(
        movies_table[["movieId", "title", "display_title", "genres", "release_year"]],
        on="movieId",
        how="left",
    )


def build_popularity_table(recommender) -> pd.DataFrame:
    """Precompute the popularity ranking."""
    popularity = recommender.recommend(top_k=5000, min_ratings=1).copy()
    popularity["rank"] = range(1, len(popularity) + 1)
    return popularity


def main() -> None:
    """Build the lightweight public-serving artifact."""
    bundle = joblib.load(SOURCE_BUNDLE_PATH)
    movies_table = build_movies_table(bundle["movies"])

    public_bundle = {
        "dataset_name": "timdb-bollywood",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_n": TOP_N,
        "metrics": bundle["metrics"]["collaborative"],
        "movies": movies_table,
        "users": sorted(bundle["ratings"]["userId"].unique().tolist()),
        "popularity": build_popularity_table(bundle["recommenders"]["popularity"]),
        "content": build_content_table(bundle["recommenders"]["content"], movies_table),
        "collaborative": build_collaborative_table(
            bundle["recommenders"]["collaborative"],
            bundle["ratings"],
            movies_table,
        ),
    }

    joblib.dump(public_bundle, OUTPUT_PATH, compress=3)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
