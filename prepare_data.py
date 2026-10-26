
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading datasets...")

ratings = pd.read_csv(DATA_DIR / "ratings_small.csv")
links = pd.read_csv(DATA_DIR / "links.csv")

movies = pd.read_csv(
    DATA_DIR / "movies_metadata.csv",
    low_memory=False
)

# Clean and standardize IDs
ratings["movieId"] = pd.to_numeric(
    ratings["movieId"], errors="coerce"
)
links["movieId"] = pd.to_numeric(
    links["movieId"], errors="coerce"
)
links["tmdbId"] = pd.to_numeric(
    links["tmdbId"], errors="coerce"
)
movies["id"] = pd.to_numeric(
    movies["id"], errors="coerce"
)

ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
).copy()

links = links.dropna(
    subset=["movieId", "tmdbId"]
).copy()

movies = movies.dropna(
    subset=["id", "title"]
).copy()

ratings["movieId"] = ratings["movieId"].astype("int64")
links["movieId"] = links["movieId"].astype("int64")
links["tmdbId"] = links["tmdbId"].astype("int64")
movies["id"] = movies["id"].astype("int64")

# Remove duplicate mapping and metadata IDs
links = links.drop_duplicates(subset=["movieId"])
movies = movies.drop_duplicates(subset=["id"])

# Map ratings to TMDB IDs, then join movie metadata
ratings_with_links = ratings.merge(
    links[["movieId", "tmdbId"]],
    on="movieId",
    how="inner"
)

clean_movies = movies[
    ["id", "title", "genres", "overview",
     "vote_average", "vote_count", "release_date"]
].copy()

clean_movies = clean_movies.rename(columns={"id": "tmdbId"})

movie_ratings = ratings_with_links.merge(
    clean_movies,
    on="tmdbId",
    how="inner"
)

# Save the prepared data
ratings.to_csv(
    OUTPUT_DIR / "clean_ratings.csv", index=False
)
clean_movies.to_csv(
    OUTPUT_DIR / "clean_movies.csv", index=False
)
movie_ratings.to_csv(
    OUTPUT_DIR / "ratings_with_movies.csv", index=False
)

print("\nDATA PREPARATION REPORT")
print("Original ratings:", len(ratings))
print("Ratings linked to TMDB:", len(ratings_with_links))
print("Clean movie records:", len(clean_movies))
print("Ratings joined with metadata:", len(movie_ratings))
print("Unique users:", movie_ratings["userId"].nunique())
print("Unique movies joined:", movie_ratings["movieId"].nunique())
print("\nSaved files in the outputs folder.")
