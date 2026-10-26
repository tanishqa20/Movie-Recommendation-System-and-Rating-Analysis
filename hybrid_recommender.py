
import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("outputs/clean_movies.csv")
ratings = pd.read_csv("outputs/clean_ratings.csv")
links = pd.read_csv("data/links.csv")

# Prepare movie genres and overview
def parse_genres(value):
    try:
        genres = ast.literal_eval(value)
        return " ".join(
            item["name"].lower()
            for item in genres
            if isinstance(item, dict) and "name" in item
        )
    except (ValueError, SyntaxError, TypeError):
        return ""

movies["genres_text"] = movies["genres"].apply(parse_genres)
movies["overview_text"] = movies["overview"].fillna("").astype(str)
movies["title"] = movies["title"].fillna("").astype(str)

# Content-based features
movies["content_text"] = (
    movies["genres_text"] + " " + movies["overview_text"]
).str.strip()

vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
content_matrix = vectorizer.fit_transform(movies["content_text"])

# Prepare collaborative-filtering data
ratings["movieId"] = pd.to_numeric(
    ratings["movieId"], errors="coerce"
)
ratings["rating"] = pd.to_numeric(
    ratings["rating"], errors="coerce"
)
ratings = ratings.dropna(subset=["movieId", "rating"])
ratings["movieId"] = ratings["movieId"].astype(int)

movie_counts = ratings.groupby("movieId")["rating"].count()
popular_ids = movie_counts[movie_counts >= 10].index
filtered_ratings = ratings[ratings["movieId"].isin(popular_ids)]

rating_matrix = filtered_ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

item_similarity = cosine_similarity(rating_matrix.fillna(0).T)

movie_ids = rating_matrix.columns.tolist()
collab_index = {
    movie_id: index for index, movie_id in enumerate(movie_ids)
}

# Map MovieLens movie IDs to TMDB IDs and titles
links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce")
links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
links = links.dropna(subset=["movieId", "tmdbId"])
links["movieId"] = links["movieId"].astype(int)
links["tmdbId"] = links["tmdbId"].astype(int)

movie_lookup = movies.drop_duplicates("tmdbId").set_index("tmdbId")
links = links[links["tmdbId"].isin(movie_lookup.index)]

movie_to_tmdb = links.set_index("movieId")["tmdbId"].to_dict()
tmdb_to_movie = links.drop_duplicates("tmdbId").set_index(
    "tmdbId"
)["movieId"].to_dict()

# Normalize scores to a 0-1 range
def normalize_scores(series):
    if series.empty:
        return series
    minimum = series.min()
    maximum = series.max()
    if maximum == minimum:
        return pd.Series(1.0, index=series.index)
    return (series - minimum) / (maximum - minimum)

# Hybrid recommender
def recommend_movies(movie_title, top_n=10):
    matches = movies[
        movies["title"].str.casefold() == movie_title.strip().casefold()
    ]

    if matches.empty:
        suggestions = movies[
            movies["title"].str.contains(
                movie_title.strip(), case=False, na=False, regex=False
            )
        ]["title"].drop_duplicates().head(10).tolist()

        print(f"\nMovie not found: {movie_title}")
        if suggestions:
            print("Possible matches:", suggestions)
        else:
            print("No similar titles found.")
        return pd.DataFrame()

    source = matches.iloc[0]
    source_tmdb = int(source["tmdbId"])
    source_title = source["title"]

    # Content-based scores
    source_index = source.name
    content_scores = cosine_similarity(
        content_matrix[source_index], content_matrix
    ).flatten()

    content_results = pd.DataFrame({
        "tmdbId": movies["tmdbId"].values,
        "content_score": content_scores
    })

    content_results = content_results[
        content_results["tmdbId"] != source_tmdb
    ]
    content_results = content_results.drop_duplicates("tmdbId")
    content_results = content_results.nlargest(50, "content_score")
    content_results["content_score"] = normalize_scores(
        content_results["content_score"]
    )

    # Collaborative scores
    source_movie_id = tmdb_to_movie.get(source_tmdb)
    collab_results = pd.DataFrame(
        columns=["tmdbId", "collaborative_score"]
    )

    if source_movie_id in collab_index:
        source_collab_index = collab_index[source_movie_id]
        scores = item_similarity[source_collab_index]

        collab_results = pd.DataFrame({
            "movieId": movie_ids,
            "collaborative_score": scores
        })

        collab_results = collab_results[
            collab_results["movieId"] != source_movie_id
        ]
        collab_results["tmdbId"] = collab_results["movieId"].map(
            movie_to_tmdb
        )
        collab_results = collab_results.dropna(subset=["tmdbId"])
        collab_results["tmdbId"] = collab_results["tmdbId"].astype(int)
        collab_results = collab_results[
            collab_results["tmdbId"].isin(movie_lookup.index)
        ]
        collab_results = collab_results.drop_duplicates("tmdbId")
        collab_results = collab_results.nlargest(
            50, "collaborative_score"
        )
        collab_results["collaborative_score"] = normalize_scores(
            collab_results["collaborative_score"]
        )
        collab_results = collab_results[
            ["tmdbId", "collaborative_score"]
        ]

    # Combine both recommendation lists
    hybrid = content_results.merge(
        collab_results, on="tmdbId", how="outer"
    )

    hybrid["content_score"] = hybrid["content_score"].fillna(0)
    hybrid["collaborative_score"] = hybrid[
        "collaborative_score"
    ].fillna(0)

    # Initial hybrid weights: content 60%, collaborative 40%
    hybrid["hybrid_score"] = (
        0.60 * hybrid["content_score"]
        + 0.40 * hybrid["collaborative_score"]
    )

    # Attach movie details
    hybrid = hybrid.merge(
        movies[
            ["tmdbId", "title", "genres_text", "vote_average"]
        ].drop_duplicates("tmdbId"),
        on="tmdbId",
        how="left"
    )

    hybrid = hybrid[hybrid["tmdbId"] != source_tmdb]
    hybrid = hybrid.sort_values("hybrid_score", ascending=False)

    result = hybrid[
        [
            "title",
            "genres_text",
            "vote_average",
            "content_score",
            "collaborative_score",
            "hybrid_score"
        ]
    ].head(top_n).reset_index(drop=True)

    print(f"\nHybrid recommendations for: {source_title}")
    print("-" * 85)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return result


if __name__ == "__main__":
    recommend_movies("Toy Story", top_n=10)
