"""Public Streamlit app for Bollywood recommendations."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "bollywood_public_bundle.joblib"


@st.cache_resource
def load_bundle() -> dict:
    """Load the lightweight public artifact."""
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "Missing bollywood_public_bundle.joblib. Run build_public_bundle.py before deployment."
        )
    return joblib.load(BUNDLE_PATH)


def render_table(df: pd.DataFrame) -> None:
    """Display a table in the app."""
    st.dataframe(df.reset_index(drop=True), width="stretch", hide_index=True)


def popularity_view(bundle: dict, top_k: int) -> None:
    """Render popularity recommendations."""
    st.subheader("Top Bollywood Movies")
    min_ratings = st.slider("Minimum ratings", min_value=10, max_value=300, value=50, step=10)
    popularity = bundle["popularity"]
    recommendations = popularity.loc[popularity["rating_count"] >= min_ratings].head(top_k).copy()
    render_table(recommendations[["title", "genres", "avg_rating", "rating_count", "weighted_score"]])


def content_view(bundle: dict, top_k: int) -> None:
    """Render content-based recommendations."""
    st.subheader("Find Similar Bollywood Movies")
    movies = bundle["movies"].sort_values("display_title")
    display_title = st.selectbox("Pick a Bollywood movie", movies["display_title"].tolist())
    anchor_movie_id = int(movies.loc[movies["display_title"] == display_title, "movieId"].iloc[0])
    recommendations = bundle["content"].loc[bundle["content"]["anchor_movieId"] == anchor_movie_id].head(top_k).copy()
    render_table(recommendations[["display_title", "genres", "content_score"]].rename(columns={"display_title": "title"}))


def collaborative_view(bundle: dict, top_k: int) -> None:
    """Render collaborative recommendations."""
    st.subheader("Personalized Bollywood Recommendations")
    users = bundle["users"]
    user_id = st.number_input(
        "User ID",
        min_value=int(min(users)),
        max_value=int(max(users)),
        value=int(users[0]),
        step=1,
    )
    collaborative = bundle["collaborative"]
    recommendations = collaborative.loc[collaborative["userId"] == int(user_id)].head(top_k).copy()
    render_table(
        recommendations[["display_title", "genres", "predicted_rating"]].rename(columns={"display_title": "title"})
    )


def hybrid_view(bundle: dict, top_k: int) -> None:
    """Render hybrid recommendations by combining public tables."""
    st.subheader("Hybrid Bollywood Recommendations")
    users = bundle["users"]
    user_id = st.number_input(
        "User ID",
        min_value=int(min(users)),
        max_value=int(max(users)),
        value=int(users[0]),
        step=1,
        key="hybrid_user_id",
    )
    movies = bundle["movies"].sort_values("display_title")
    display_title = st.selectbox("Anchor movie", movies["display_title"].tolist(), key="hybrid_title")
    anchor_movie_id = int(movies.loc[movies["display_title"] == display_title, "movieId"].iloc[0])

    collaborative = bundle["collaborative"].loc[bundle["collaborative"]["userId"] == int(user_id)].copy()
    content = bundle["content"].loc[bundle["content"]["anchor_movieId"] == anchor_movie_id].copy()

    hybrid = collaborative.merge(
        content[["movieId", "content_score"]],
        on="movieId",
        how="outer",
    )
    hybrid["userId"] = int(user_id)
    hybrid["predicted_rating"] = hybrid["predicted_rating"].fillna(0.0)
    hybrid["collaborative_score"] = hybrid["collaborative_score"].fillna(0.0)
    hybrid["content_score"] = hybrid["content_score"].fillna(0.0)
    hybrid = hybrid.merge(
        bundle["movies"][["movieId", "display_title", "genres"]],
        on="movieId",
        how="left",
    )
    hybrid = hybrid.loc[hybrid["movieId"] != anchor_movie_id].copy()
    hybrid["hybrid_score"] = 0.7 * hybrid["collaborative_score"] + 0.3 * hybrid["content_score"]
    hybrid = hybrid.sort_values("hybrid_score", ascending=False).head(top_k)
    render_table(hybrid[["display_title", "genres", "hybrid_score"]].rename(columns={"display_title": "title"}))


def metrics_view(bundle: dict) -> None:
    """Render model metrics."""
    st.subheader("Bollywood Model Metrics")
    metrics = pd.DataFrame(bundle["metrics"]).T.reset_index().rename(columns={"index": "model"})
    render_table(metrics)


def main() -> None:
    """Run the public app."""
    st.set_page_config(page_title="Bollywood Movie Recommender", layout="wide")
    st.title("Bollywood Movie Recommender")
    st.caption("Public-ready Bollywood recommendations powered by TIMDB metadata and precomputed collaborative signals.")

    bundle = load_bundle()
    st.caption(f"Dataset: `{bundle['dataset_name']}` | Generated: `{bundle['generated_at_utc']}`")

    st.sidebar.header("Controls")
    top_k = st.sidebar.slider("Top K", min_value=5, max_value=20, value=10)
    mode = st.sidebar.selectbox(
        "Recommendation mode",
        ["Popularity", "Content-Based", "Collaborative SVD", "Hybrid", "Metrics"],
    )

    if mode == "Popularity":
        popularity_view(bundle, top_k)
    elif mode == "Content-Based":
        content_view(bundle, top_k)
    elif mode == "Collaborative SVD":
        collaborative_view(bundle, top_k)
    elif mode == "Hybrid":
        hybrid_view(bundle, top_k)
    else:
        metrics_view(bundle)


if __name__ == "__main__":
    main()
