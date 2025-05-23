import pandas as pd
import os

DATA_DIR = "data"
MOVIES_FILE = os.path.join(DATA_DIR, "movies.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
NEW_RATINGS_FILE = os.path.join(DATA_DIR, "new_ratings.csv")

def load_movies():
    df = pd.read_csv(
        MOVIES_FILE,
        sep="|",
        encoding="latin1",
        header=None,
        names=[
            "movieId", "title", "release_date", "video_release_date", "IMDb_URL",
            "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
    )
    genre_columns = df.columns[5:]  # all genre columns
    df["genres"] = df[genre_columns].apply(lambda row: "|".join(genre_columns[row == 1]), axis=1)
    return df[["movieId", "title", "genres"]]


def load_ratings():
    base_ratings = pd.read_csv(
        RATINGS_FILE,
        sep="\t",  # tab-delimited
        header=None,
        names=["userId", "movieId", "rating", "timestamp"]
    )
    if os.path.exists(NEW_RATINGS_FILE):
        new_ratings = pd.read_csv(NEW_RATINGS_FILE)
        base_ratings = pd.concat([base_ratings, new_ratings], ignore_index=True)
    return base_ratings[["userId", "movieId", "rating"]]


def save_new_ratings(new_ratings_df):
    if os.path.exists(NEW_RATINGS_FILE):
        existing = pd.read_csv(NEW_RATINGS_FILE)
        combined = pd.concat([existing, new_ratings_df], ignore_index=True)
    else:
        combined = new_ratings_df
    combined.to_csv(NEW_RATINGS_FILE, index=False)
