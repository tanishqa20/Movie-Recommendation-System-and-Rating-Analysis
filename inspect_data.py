
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

ratings = pd.read_csv(DATA_DIR / "ratings_small.csv")
movies = pd.read_csv(DATA_DIR / "movies_metadata.csv", low_memory=False)
links = pd.read_csv(DATA_DIR / "links.csv")

print("\n========== CINEMATCH DATASET REPORT ==========")

print("\n1. RATINGS DATA")
print("Rows:", len(ratings))
print("Columns:", ratings.columns.tolist())
print("Unique users:", ratings["userId"].nunique())
print("Unique movies rated:", ratings["movieId"].nunique())
print("Missing values:\n", ratings.isnull().sum())

print("\n2. MOVIE METADATA")
print("Rows:", len(movies))
print("Columns:", movies.columns.tolist())
print("Missing titles:", movies["title"].isnull().sum())
print("Missing overviews:", movies["overview"].isnull().sum())

print("\n3. MOVIE LINKS")
print("Rows:", len(links))
print("Columns:", links.columns.tolist())

print("\nSample ratings:")
print(ratings.head())

print("\nSample movies:")
print(movies[["id", "title", "genres", "vote_average"]].head())

print("\nDataset inspection completed.")
