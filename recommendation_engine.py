
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent

# Support both the repository root and the outputs folder.
possible_paths = [
    BASE_DIR / "outputs" / "clean_movies.csv",
    BASE_DIR / "clean_movies.csv",
]

MOVIES_PATH = next((p for p in possible_paths if p.is_file()), None)

if MOVIES_PATH is None:
    raise FileNotFoundError(
        "clean_movies.csv was not found. "
        "Place it in the repository root or the outputs folder."
    )

movies = pd.read_csv(MOVIES_PATH)

required_columns = {"title", "genres", "overview", "vote_average"}
missing_columns = required_columns - set(movies.columns)

if missing_columns:
    raise ValueError(
        f"clean_movies.csv is missing columns: {sorted(missing_columns)}"
    )

movies = movies.dropna(subset=["title"]).copy()

for column in ["genres", "overview"]:
    movies[column] = movies[column].fillna("").astype(str)

movies["vote_average"] = pd.to_numeric(
    movies["vote_average"], errors="coerce"
).fillna(0)

movies["combined_features"] = (
    movies["genres"] + " " + movies["overview"]
).str.strip()

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=20000
)

tfidf_matrix = tfidf.fit_transform(movies["combined_features"])


def recommend_movies(title, top_n=10):
    matches = movies[
        movies["title"].str.casefold() == str(title).casefold()
    ]

    if matches.empty:
        partial_matches = movies[
            movies["title"].str.contains(
                str(title), case=False, na=False, regex=False
            )
        ]
        if partial_matches.empty:
            return pd.DataFrame()
        movie_index = partial_matches.index[0]
    else:
        movie_index = matches.index[0]

    position = movies.index.get_loc(movie_index)

    scores = cosine_similarity(
        tfidf_matrix[position], tfidf_matrix
    ).ravel()

    ranked_positions = scores.argsort()[::-1]
    ranked_positions = [
        i for i in ranked_positions if i != position
    ][:top_n]

    recommendations = movies.iloc[ranked_positions].copy()
    recommendations["similarity_score"] = scores[ranked_positions]

    return recommendations[
        ["title", "genres", "vote_average", "similarity_score"]
    ].reset_index(drop=True)
