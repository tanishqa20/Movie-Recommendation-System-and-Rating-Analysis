
import streamlit as st
import pandas as pd
import zipfile
from pathlib import Path
from io import BytesIO
from PIL import Image
import re
import inspect
import time

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# DARK CINEMATIC THEME
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at top left, #252044 0%, transparent 42%),
        radial-gradient(ellipse at top right, #361b2d 0%, transparent 35%),
        linear-gradient(135deg, #090b13 0%, #11111e 50%, #090b12 100%);
    color: #f4f4f8;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: rgba(9, 11, 19, 0.85);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1, h2, h3, p, label, .stMarkdown {
    color: #f4f4f8;
}

.hero {
    padding: 34px 38px;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 22px;
    background: linear-gradient(
        120deg,
        rgba(35, 31, 58, 0.96),
        rgba(28, 22, 39, 0.88),
        rgba(54, 26, 43, 0.82)
    );
    box-shadow: 0 15px 45px rgba(0,0,0,0.25);
    margin-bottom: 28px;
}

.hero h1 {
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 8px;
}

.hero p {
    color: #d0cde0;
    font-size: 1rem;
    line-height: 1.7;
}

.brand-accent {
    color: #c4a7ff;
}

.section-heading {
    font-size: 1.35rem;
    font-weight: 700;
    margin: 20px 0 14px 0;
    color: #f4f0ff;
}

/* Dropdown and number input labels */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #e9ddff !important;
    font-weight: 600 !important;
}

/* Dropdown selected value */
div[data-baseweb="select"] > div {
    background-color: #202033 !important;
    color: #ffffff !important;
    border: 1px solid #65558e !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

div[data-baseweb="select"] span {
    color: #ffffff !important;
}

/* Dropdown menu */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #202033 !important;
}

li[role="option"],
div[role="option"] {
    background-color: #202033 !important;
    color: #ffffff !important;
}

li[role="option"]:hover,
div[role="option"]:hover {
    background-color: #49366b !important;
    color: #ffffff !important;
}

/* Number input */
div[data-testid="stNumberInput"] input {
    background-color: #202033 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #8d72ce !important;
    border-radius: 8px !important;
}

/* Recommendation button */
.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 11px;
    border: 1px solid #b39aff;
    color: #ffffff !important;
    font-weight: 700;
    background: linear-gradient(100deg, #7655d9, #a24e9a);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #e0d4ff;
    background: linear-gradient(100deg, #8968ed, #bd5bad);
    transform: translateY(-1px);
    box-shadow: 0 7px 20px rgba(130, 83, 220, 0.25);
}

/* Movie cards */
.movie-title {
    color: #f8f6ff !important;
    font-size: 0.98rem;
    font-weight: 700;
    margin: 10px 0 5px 0;
    line-height: 1.4;
    overflow-wrap: anywhere;
}

.movie-meta {
    color: #c1bdd5 !important;
    font-size: 0.8rem;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

.poster-placeholder {
    height: 250px;
    border: 1px dashed #49445f;
    border-radius: 10px;
    background: linear-gradient(150deg, #242238, #151522);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: #c8bddf;
    text-align: center;
    padding: 12px;
}

div[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.10);
}

/* Cinematic loading message */
div[data-testid="stSpinner"] {
    color: #d7c4ff !important;
    font-weight: 600;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #2b2b40;
}

.footer {
    text-align: center;
    color: #9692ad;
    font-size: 0.82rem;
    padding: 25px 0 5px 0;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


CLEAN_MOVIES = OUTPUT_DIR / "clean_movies.csv"
if not CLEAN_MOVIES.exists():
    CLEAN_MOVIES = BASE_DIR / "clean_movies.csv"

METADATA_FILE = DATA_DIR / "movies_metadata.csv"
LINKS_FILE = DATA_DIR / "links.csv"

# --------------------------------------------------
# FIND POSTERLENS ZIP
# --------------------------------------------------

def find_poster_zip():
    search_dirs = [
        BASE_DIR,
        BASE_DIR.parent,
        BASE_DIR.parent.parent
    ]

    for folder in search_dirs:
        if not folder.exists():
            continue

        for zip_path in folder.glob("*.zip"):
            try:
                with zipfile.ZipFile(zip_path, "r") as archive:
                    names = archive.namelist()

                    if any(
                        "/covers/" in name.lower()
                        or name.lower().startswith("covers/")
                        for name in names
                    ):
                        return zip_path

            except (zipfile.BadZipFile, OSError):
                continue

    return None


ZIP_PATH = find_poster_zip()

# --------------------------------------------------
# INDEX POSTERS WITHOUT EXTRACTING
# --------------------------------------------------

@st.cache_data(show_spinner="🎞️ Loading movie poster library...")
def index_covers(zip_path):
    cover_index = {}

    if not zip_path:
        return cover_index

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for name in archive.namelist():
                lower_name = name.lower()

                if (
                    "/covers/" not in lower_name
                    and not lower_name.startswith("covers/")
                ):
                    continue

                if not lower_name.endswith(
                    (".jpg", ".jpeg", ".png", ".webp")
                ):
                    continue

                cover_id = Path(name).stem.strip().lower()

                if cover_id:
                    cover_index[cover_id] = name

    except (zipfile.BadZipFile, OSError):
        pass

    return cover_index


cover_paths = index_covers(str(ZIP_PATH) if ZIP_PATH else None)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def read_csv_safe(path):
    if path.exists():
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


movies = read_csv_safe(CLEAN_MOVIES)
metadata = read_csv_safe(METADATA_FILE)
links = read_csv_safe(LINKS_FILE)

# --------------------------------------------------
# NORMALIZE TITLES AND IDS
# --------------------------------------------------

def normalize_title(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-z0-9 ]", "", value)

    return value


def normalize_imdb_id(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    if value in ("", "nan", "none"):
        return ""

    if value.startswith("tt"):
        value = value[2:]

    if value.endswith(".0"):
        value = value[:-2]

    if value.isdigit():
        return value.zfill(7)

    return value


def normalize_tmdb_id(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value in ("", "nan", "None"):
        return ""

    if value.endswith(".0"):
        value = value[:-2]

    return value

# --------------------------------------------------
# BUILD MOVIE LOOKUP
# --------------------------------------------------

def build_movie_lookup(movies_df, metadata_df, links_df):
    result = movies_df.copy()

    if result.empty:
        return result

    if "title" not in result.columns:
        result["title"] = (
            result["original_title"]
            if "original_title" in result.columns
            else ""
        )

    result["title_key"] = result["title"].apply(normalize_title)

    if not links_df.empty and "movieId" in links_df.columns:
        link_columns = ["movieId"]

        for column in ["imdbId", "tmdbId"]:
            if column in links_df.columns:
                link_columns.append(column)

        usable_links = links_df[link_columns].copy()

        if "movieId" in result.columns:
            result["movieId"] = result["movieId"].astype(str).str.replace(
                r"\.0$", "", regex=True
            )

            usable_links["movieId"] = usable_links["movieId"].astype(
                str
            ).str.replace(r"\.0$", "", regex=True)

            result = result.merge(
                usable_links.drop_duplicates("movieId"),
                on="movieId",
                how="left",
                suffixes=("", "_link")
            )

        elif (
            "tmdbId" in result.columns
            and "tmdbId" in usable_links.columns
        ):
            result["tmdbId"] = result["tmdbId"].apply(normalize_tmdb_id)
            usable_links["tmdbId"] = usable_links["tmdbId"].apply(
                normalize_tmdb_id
            )

            result = result.merge(
                usable_links.drop_duplicates("tmdbId"),
                on="tmdbId",
                how="left",
                suffixes=("", "_link")
            )

    title_to_imdb = {}

    if not metadata_df.empty and "title" in metadata_df.columns:
        imdb_column = next(
            (
                col for col in metadata_df.columns
                if col.lower() in ("imdb_id", "imdbid")
            ),
            None
        )

        if imdb_column:
            meta = metadata_df[["title", imdb_column]].copy()
            meta["title_key"] = meta["title"].apply(normalize_title)
            meta["poster_imdb_id"] = meta[imdb_column].apply(
                normalize_imdb_id
            )

            title_to_imdb = (
                meta.drop_duplicates("title_key")
                .set_index("title_key")["poster_imdb_id"]
                .to_dict()
            )

    if "imdbId" in result.columns:
        result["poster_imdb_id"] = result["imdbId"].apply(
            normalize_imdb_id
        )
    else:
        result["poster_imdb_id"] = ""

    result["poster_imdb_id"] = result.apply(
        lambda row: row["poster_imdb_id"]
        or title_to_imdb.get(row["title_key"], ""),
        axis=1
    )

    return result

movie_lookup = build_movie_lookup(movies, metadata, links)

# --------------------------------------------------
# MATCH RECOMMENDATIONS WITH POSTERS
# --------------------------------------------------

def attach_posters(results):
    output = results.copy()

    if output.empty:
        return output

    if "title" not in output.columns:
        output["title"] = ""

    output["title_key"] = output["title"].apply(normalize_title)

    if (
        not movie_lookup.empty
        and "poster_imdb_id" in movie_lookup.columns
    ):
        lookup = movie_lookup.drop_duplicates("title_key")

        title_to_id = lookup.set_index("title_key")["poster_imdb_id"]

        output["poster_imdb_id"] = output["title_key"].map(
            title_to_id
        ).fillna("")
    else:
        output["poster_imdb_id"] = ""

    output["local_cover"] = output["poster_imdb_id"].apply(
        lambda value: cover_paths.get(normalize_imdb_id(value), "")
    )

    return output


@st.cache_data(show_spinner=False)
def read_cover(zip_path, internal_path):
    if not zip_path or not internal_path:
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            image_bytes = archive.read(internal_path)

        image = Image.open(BytesIO(image_bytes))
        image.load()

        return image.copy()

    except Exception:
        return None

# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🎬 Cine<span class="brand-accent">Match</span></h1>
    <p>
        Your next favourite movie is just one recommendation away.
        Explore films selected through content-based, collaborative,
        and hybrid recommendation techniques.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD RECOMMENDATION ENGINE
# --------------------------------------------------

try:
    from recommendation_engine import recommend_movies
except Exception as error:
    recommend_movies = None
    st.error("Could not load the recommendation engine.")
    st.exception(error)

# --------------------------------------------------
# MOVIE SELECTION
# --------------------------------------------------

if movies.empty:
    st.error(
        "Movie data could not be loaded. Check "
        "outputs/clean_movies.csv."
    )

elif recommend_movies is None:
    st.warning(
        "Please check recommendation_engine.py and "
        "make sure it defines recommend_movies."
    )

elif "title" not in movies.columns:
    st.error("The movie dataset does not contain a title column.")

else:
    st.markdown(
        '<div class="section-heading">Find your next movie</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.3, 1])

    with left:
        movie_titles = (
            movies["title"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        selected_movie = st.selectbox(
            "Select a movie you already like",
            movie_titles,
            help="Choose a movie to discover similar films."
        )

    with right:
        recommendation_type = st.selectbox(
            "Recommendation method",
            [
                "Hybrid",
                "Content-Based",
                "Collaborative Filtering"
            ]
        )

    number_of_recommendations = st.number_input(
        "Number of recommendations",
        min_value=4,
        max_value=16,
        value=8,
        step=1,
        format="%d",
        help="Enter a number between 4 and 16."
    )

    run_recommendation = st.button(
        "✨  Find My Movies",
        type="primary"
    )

    if run_recommendation:
        # Show an animated spinner while the model runs.
        with st.spinner(
            "🎬 Rolling the film reel... Finding movies you'll love"
        ):
            try:
                parameters = inspect.signature(
                    recommend_movies
                ).parameters

                possible_args = {
                    "movie_title": selected_movie,
                    "title": selected_movie,
                    "selected_movie": selected_movie,
                    "movie_name": selected_movie,
                    "method": recommendation_type,
                    "recommendation_type": recommendation_type,
                    "model_type": recommendation_type,
                    "n_recommendations": int(number_of_recommendations),
                    "num_recommendations": int(number_of_recommendations),
                    "n": int(number_of_recommendations),
                    "top_n": int(number_of_recommendations),
                    "top_n_recommendations": int(number_of_recommendations)
                }

                kwargs = {
                    key: value
                    for key, value in possible_args.items()
                    if key in parameters
                }

                if kwargs:
                    recommendations = recommend_movies(**kwargs)
                else:
                    recommendations = recommend_movies(selected_movie)

                if isinstance(recommendations, tuple):
                    recommendations = recommendations[0]

                if not isinstance(recommendations, pd.DataFrame):
                    recommendations = pd.DataFrame(recommendations)

                if not recommendations.empty:
                    recommendations = attach_posters(recommendations)

            except Exception as error:
                recommendations = None
                st.error(
                    "Something went wrong while generating recommendations."
                )
                st.exception(error)

        # Display results after the recommendation model finishes.
        if recommendations is not None:
            if recommendations.empty:
                st.warning("No recommendations were found.")
            else:
                st.markdown(
                    '<div class="section-heading">🍿 Recommended for you</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    f"Based on your selection: {selected_movie}"
                )

                display_results = recommendations.head(
                    int(number_of_recommendations)
                )

                columns = st.columns(4)

                for index, (_, movie) in enumerate(
                    display_results.iterrows()
                ):
                    with columns[index % 4]:
                        title = str(
                            movie.get("title", "Unknown title")
                        )

                        poster_path = movie.get("local_cover", "")
                        poster_image = None

                        if (
                            isinstance(poster_path, str)
                            and poster_path
                            and ZIP_PATH
                        ):
                            poster_image = read_cover(
                                str(ZIP_PATH),
                                poster_path
                            )

                        if poster_image is not None:
                            st.image(
                                poster_image,
                                use_container_width=True
                            )
                        else:
                            st.markdown("""
                            <div class="poster-placeholder">
                                <div style="font-size:2rem;">🎞️</div>
                                <div style="margin-top:10px;">
                                    Poster unavailable
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown(
                            f'<div class="movie-title">{title}</div>',
                            unsafe_allow_html=True
                        )

                        if "genres_text" in movie.index:
                            st.markdown(
                                f'<div class="movie-meta">'
                                f'{movie["genres_text"]}</div>',
                                unsafe_allow_html=True
                            )

                        if "vote_average" in movie.index:
                            try:
                                rating = float(movie["vote_average"])

                                st.markdown(
                                    f'<div class="movie-meta">'
                                    f'⭐ Rating: {rating:.1f}</div>',
                                    unsafe_allow_html=True
                                )
                            except (ValueError, TypeError):
                                pass

                        if "hybrid_score" in movie.index:
                            try:
                                score = float(movie["hybrid_score"])

                                st.markdown(
                                    f'<div class="movie-meta">'
                                    f'Match score: {score:.3f}</div>',
                                    unsafe_allow_html=True
                                )
                            except (ValueError, TypeError):
                                pass

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<hr>
<div class="footer">
    CineMatch · Discover movies you'll love 🎬
</div>
""", unsafe_allow_html=True)
