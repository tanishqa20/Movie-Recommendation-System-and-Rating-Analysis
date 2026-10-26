
import pandas as pd
from pathlib import Path

data = Path("data")

ratings = pd.read_csv(data / "ratings_small.csv")
movies = pd.read_csv(
    data / "movies_metadata.csv",
    low_memory=False
)
links = pd.read_csv(data / "links.csv")

print("RATINGS")
print(ratings[["userId", "movieId", "rating"]].head())

print("\nMOVIE METADATA")
print(movies[["id", "title"]].head())

print("\nLINKS")
print(links.head())

# Convert metadata IDs to numeric values safely
movies["id"] = pd.to_numeric(movies["id"], errors="coerce")

valid_movies = movies.dropna(subset=["id"]).copy()
valid_movies["id"] = valid_movies["id"].astype("int64")

rating_ids = set(ratings["movieId"].unique())
metadata_ids = set(valid_movies["id"].unique())
link_ids = set(links["movieId"].unique())

print("\nID MATCHING REPORT")
print("Unique rated movie IDs:", len(rating_ids))
print("Metadata IDs:", len(metadata_ids))
print("Direct matches:", len(rating_ids & metadata_ids))
print("Ratings IDs present in links:", len(rating_ids & link_ids))

print("\nID check completed.")
