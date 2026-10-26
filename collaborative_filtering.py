
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("outputs")

# Load real ratings and cleaned movie metadata
ratings = pd.read_csv(DATA_DIR / "clean_ratings.csv")
movies = pd.read_csv(DATA_DIR / "clean_movies.csv")
links = pd.read_csv("data/links.csv")

# Clean ratings
ratings = ratings[["userId", "movieId", "rating"]].dropna()
ratings["movieId"] = pd.to_numeric(
    ratings["movieId"], errors="coerce"
)
ratings = ratings.dropna(subset=["movieId"])
ratings["movieId"] = ratings["movieId"].astype(int)

# Keep movies with at least 10 ratings
movie_counts = ratings.groupby("movieId")["rating"].count()
popular_ids = movie_counts[movie_counts >= 10].index
ratings = ratings[ratings["movieId"].isin(popular_ids)]

# Create user-movie rating matrix
rating_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

print("Users:", rating_matrix.shape[0])
print("Movies:", rating_matrix.shape[1])

# Calculate item-based collaborative similarity
rating_values = rating_matrix.fillna(0)
movie_similarity = cosine_similarity(rating_values.T)

similarity_df = pd.DataFrame(
    movie_similarity,
    index=rating_matrix.columns,
    columns=rating_matrix.columns
)

print("Collaborative filtering model created successfully.")

# Prepare MovieLens-to-TMDB title mapping
links = links.dropna(subset=["movieId", "tmdbId"]).copy()
links["movieId"] = pd.to_numeric(
    links["movieId"], errors="coerce"
)
links["tmdbId"] = pd.to_numeric(
    links["tmdbId"], errors="coerce"
)
links = links.dropna(subset=["movieId", "tmdbId"])

links["movieId"] = links["movieId"].astype(int)
links["tmdbId"] = links["tmdbId"].astype(int)

movies["tmdbId"] = pd.to_numeric(
    movies["tmdbId"], errors="coerce"
)
movies = movies.dropna(subset=["tmdbId", "title"]).copy()
movies["tmdbId"] = movies["tmdbId"].astype(int)

movie_lookup = links.merge(
    movies[["tmdbId", "title"]],
    on="tmdbId",
    how="left"
)

movie_lookup = movie_lookup.drop_duplicates("movieId")
movie_lookup = movie_lookup.set_index("movieId")["title"]


# Generate recommendations
def recommend_movies(movie_id, top_n=10):
    movie_id = int(movie_id)

    if movie_id not in similarity_df.columns:
        print(f"Movie ID {movie_id} is unavailable in the model.")
        return pd.DataFrame()

    scores = similarity_df[movie_id].drop(movie_id)
    scores = scores.sort_values(ascending=False).head(top_n)

    results = pd.DataFrame({
        "movieId": scores.index.astype(int),
        "similarity": scores.values.round(3)
    })

    results["title"] = results["movieId"].map(movie_lookup)

    return results[
        ["movieId", "title", "similarity"]
    ].reset_index(drop=True)


if __name__ == "__main__":
    # MovieLens movie ID 1 corresponds to Toy Story
    print("\nRecommendations for Toy Story:\n")

    results = recommend_movies(1, 10)
    print(results.to_string(index=False))

    results.to_csv(
        DATA_DIR / "collaborative_recommendations.csv",
        index=False
    )

    print(
        "\nSaved to outputs/collaborative_recommendations.csv"
    )
