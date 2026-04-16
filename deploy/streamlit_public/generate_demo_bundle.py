"""Generate a realistic demo Bollywood bundle for local testing and deployment."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).resolve().parent / "bollywood_public_bundle.joblib"

# fmt: off
BOLLYWOOD_MOVIES = [
    (1, "Dilwale Dulhania Le Jayenge", "Romance|Drama", "1995"),
    (2, "3 Idiots", "Comedy|Drama", "2009"),
    (3, "Lagaan", "Drama|Sport", "2001"),
    (4, "Dangal", "Biography|Drama|Sport", "2016"),
    (5, "PK", "Comedy|Drama|Sci-Fi", "2014"),
    (6, "Bajrangi Bhaijaan", "Action|Comedy|Drama", "2015"),
    (7, "Sholay", "Action|Adventure|Comedy", "1975"),
    (8, "Mughal-E-Azam", "Drama|Musical|Romance", "1960"),
    (9, "Dil Chahta Hai", "Comedy|Drama|Romance", "2001"),
    (10, "Rang De Basanti", "Comedy|Crime|Drama", "2006"),
    (11, "Gangs of Wasseypur", "Action|Crime|Drama", "2012"),
    (12, "Andhadhun", "Crime|Mystery|Thriller", "2018"),
    (13, "Zindagi Na Milegi Dobara", "Adventure|Comedy|Drama", "2011"),
    (14, "Barfi!", "Comedy|Drama|Romance", "2012"),
    (15, "Queen", "Adventure|Comedy|Drama", "2013"),
    (16, "Swades", "Drama", "2004"),
    (17, "Taare Zameen Par", "Drama|Family", "2007"),
    (18, "Chak De! India", "Drama|Sport", "2007"),
    (19, "Kabhi Khushi Kabhie Gham", "Drama|Musical|Romance", "2001"),
    (20, "Drishyam", "Crime|Drama|Mystery", "2015"),
    (21, "Gully Boy", "Drama|Music", "2019"),
    (22, "Tumbbad", "Fantasy|Horror|Thriller", "2018"),
    (23, "Article 15", "Crime|Drama|Thriller", "2019"),
    (24, "Panga", "Drama|Sport", "2020"),
    (25, "War", "Action|Thriller", "2019"),
    (26, "Super 30", "Biography|Drama", "2019"),
    (27, "Padmaavat", "Action|Drama|Romance", "2018"),
    (28, "Bajirao Mastani", "Action|Drama|Romance", "2015"),
    (29, "Singham", "Action|Crime|Drama", "2011"),
    (30, "Bhaag Milkha Bhaag", "Biography|Drama|Sport", "2013"),
    (31, "Raazi", "Action|Drama|Thriller", "2018"),
    (32, "Stree", "Comedy|Horror", "2018"),
    (33, "Uri: The Surgical Strike", "Action|Drama|War", "2019"),
    (34, "Kabir Singh", "Drama|Romance", "2019"),
    (35, "Chhichhore", "Comedy|Drama", "2019"),
    (36, "Tanhaji", "Action|Drama|History", "2020"),
    (37, "Rockstar", "Drama|Music|Romance", "2011"),
    (38, "Jab We Met", "Comedy|Drama|Romance", "2007"),
    (39, "Om Shanti Om", "Action|Comedy|Drama", "2007"),
    (40, "Kai Po Che!", "Drama|Sport", "2013"),
    (41, "Lootera", "Crime|Drama|Romance", "2013"),
    (42, "Haider", "Crime|Drama|Thriller", "2014"),
    (43, "Neerja", "Biography|Drama|Thriller", "2016"),
    (44, "Pink", "Crime|Drama|Thriller", "2016"),
    (45, "Newton", "Comedy|Drama", "2017"),
    (46, "Toilet: Ek Prem Katha", "Comedy|Drama|Romance", "2017"),
    (47, "Badhaai Ho", "Comedy|Drama|Family", "2018"),
    (48, "Mard Ko Dard Nahi Hota", "Action|Comedy", "2018"),
    (49, "Photograph", "Drama|Romance", "2019"),
    (50, "Ludo", "Comedy|Crime|Drama", "2020"),
]
# fmt: on

NUM_USERS = 20
TOP_N = 20
RNG = np.random.default_rng(42)


def build_movies() -> pd.DataFrame:
    rows = []
    for mid, title, genres, year in BOLLYWOOD_MOVIES:
        display_title = f"{title} ({year})" if year else title
        rows.append(
            {
                "movieId": mid,
                "title": title,
                "display_title": display_title,
                "genres": genres,
                "release_year": year,
            }
        )
    return pd.DataFrame(rows)


def build_popularity(movies: pd.DataFrame) -> pd.DataFrame:
    n = len(movies)
    pop = movies[["movieId", "title", "genres"]].copy()
    pop["avg_rating"] = np.round(RNG.uniform(2.5, 5.0, n), 2)
    pop["rating_count"] = RNG.integers(10, 500, n)
    pop["weighted_score"] = np.round(
        pop["avg_rating"] * np.log1p(pop["rating_count"]) / np.log1p(500), 2
    )
    pop = pop.sort_values("weighted_score", ascending=False).reset_index(drop=True)
    pop["rank"] = pop.index + 1
    return pop


def build_content(movies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    movie_ids = movies["movieId"].tolist()
    for anchor_id in movie_ids:
        candidates = [m for m in movie_ids if m != anchor_id]
        scores = np.sort(RNG.uniform(0, 1, len(candidates)))[::-1][:TOP_N]
        for rank, (cid, score) in enumerate(
            zip(RNG.choice(candidates, size=min(TOP_N, len(candidates)), replace=False), scores),
            start=1,
        ):
            rows.append(
                {
                    "anchor_movieId": anchor_id,
                    "movieId": int(cid),
                    "content_score": round(float(score), 4),
                    "rank": rank,
                }
            )
    content = pd.DataFrame(rows)
    content = content.merge(
        movies[["movieId", "title", "display_title", "genres", "release_year"]],
        on="movieId",
        how="left",
    )
    return content


def build_collaborative(movies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    movie_ids = movies["movieId"].tolist()
    for user_id in range(1, NUM_USERS + 1):
        preds = np.sort(RNG.uniform(1.5, 5.0, min(TOP_N, len(movie_ids))))[::-1]
        chosen = RNG.choice(movie_ids, size=min(TOP_N, len(movie_ids)), replace=False)
        for rank, (mid, pred) in enumerate(zip(chosen, preds), start=1):
            pred_val = round(float(pred), 4)
            rows.append(
                {
                    "userId": user_id,
                    "movieId": int(mid),
                    "predicted_rating": pred_val,
                    "collaborative_score": round(
                        (pred_val - 1.5) / (5.0 - 1.5), 4
                    ),
                    "rank": rank,
                }
            )
    collab = pd.DataFrame(rows)
    collab = collab.merge(
        movies[["movieId", "title", "display_title", "genres", "release_year"]],
        on="movieId",
        how="left",
    )
    return collab


def main() -> None:
    movies = build_movies()
    bundle = {
        "dataset_name": "timdb-bollywood",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_n": TOP_N,
        "metrics": {
            "svd": {"rmse": 0.87, "mae": 0.67, "precision_at_10": 0.32, "recall_at_10": 0.18},
        },
        "movies": movies,
        "users": list(range(1, NUM_USERS + 1)),
        "popularity": build_popularity(movies),
        "content": build_content(movies),
        "collaborative": build_collaborative(movies),
    }
    joblib.dump(bundle, OUTPUT_PATH, compress=3)
    print(f"Demo bundle saved to {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
