
import pandas as pd
import ast
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("outputs")

# Load movie data
movies = pd.read_csv(DATA_DIR / "clean_movies.csv")
movies = movies.dropna(subset=["title"]).copy()
movies["title"] = movies["title"].astype(str)


# Extract genre names
def extract_genres(value):
    try:
        data = ast.literal_eval(value)
        if isinstance(data, list):
            return " ".join(
                item.get("name", "")
                for item in data
                if isinstance(item, dict)
            )
    except (ValueError, SyntaxError, TypeError):
        pass
    return ""


# Prepare features
movies["genres_text"] = movies["genres"].apply(extract_genres)
movies["overview"] = movies["overview"].fillna("").astype(str)

# Remove movies without genre information
movies = movies[
    movies["genres_text"].str.strip().ne("")
].copy()

movies["genres_text"] = movies["genres_text"].str.lower()
movies["overview"] = movies["overview"].str.lower()
movies = movies.reset_index(drop=True)

# Create separate TF-IDF matrices
genre_vectorizer = TfidfVectorizer()
overview_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000
)

genre_matrix = genre_vectorizer.fit_transform(movies["genres_text"])
overview_matrix = overview_vectorizer.fit_transform(movies["overview"])

print("Content-based recommendation model created successfully.")
print("Movies available:", len(movies))


def recommend_movies(title, top_n=10):
    matches = movies[
        movies["title"].str.casefold() == title.casefold()
    ]

    if matches.empty:
        print(f"Movie not found: {title}")
        return pd.DataFrame()

    movie_index = matches.index[0]

    # Calculate similarities for the selected movie only
    genre_scores = cosine_similarity(
        genre_matrix[movie_index],
        genre_matrix
    ).flatten()

    overview_scores = cosine_similarity(
        overview_matrix[movie_index],
        overview_matrix
    ).flatten()

    # Combine genre and overview similarity
    scores = 0.70 * genre_scores + 0.30 * overview_scores

    # Rank movies by similarity
    ranked_indices = scores.argsort()[::-1]
    results = []

    for index in ranked_indices:
        if index == movie_index:
            continue

        results.append(index)

        if len(results) == top_n:
            break

    recommendations = movies.iloc[results][
        ["title", "genres_text", "vote_average"]
    ].copy()

    recommendations.insert(
        0, "similarity", scores[results].round(3)
    )

    return recommendations.reset_index(drop=True)


if __name__ == "__main__":
    for movie_title in ["Toy Story", "Jumanji"]:
        print(f"\nRecommendations for {movie_title}:\n")
        results = recommend_movies(movie_title)

        if not results.empty:
            print(results.to_string(index=False))
