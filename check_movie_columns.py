
import pandas as pd

movies = pd.read_csv("outputs/clean_movies.csv")

print("Movie columns:")
print(movies.columns.tolist())

print("\nFirst 3 rows:")
print(movies.head(3).to_string())


import pandas as pd

file_path = "data/movies_metadata.csv"

movies = pd.read_csv(file_path, low_memory=False)

print("Columns:")
print(movies.columns.tolist())

print("\nFirst 10 movie poster paths:")
print(movies[["id", "title", "poster_path"]].head(10))

print("\nTotal movies:", len(movies))
print("Available posters:", movies["poster_path"].notna().sum())
print("Missing posters:", movies["poster_path"].isna().sum())
