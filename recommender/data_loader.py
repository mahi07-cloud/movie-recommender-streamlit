import pandas as pd
import os

DATA_DIR = "data"
MOVIES_FILE = os.path.join(DATA_DIR, "movies.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
NEW_RATINGS_FILE = os.path.join(DATA_DIR, "new_ratings.csv")

def load_movies():
    return pd.read_csv(MOVIES_FILE)

def load_ratings():
    base_ratings = pd.read_csv(RATINGS_FILE)
    if os.path.exists(NEW_RATINGS_FILE):
        new_ratings = pd.read_csv(NEW_RATINGS_FILE)
        base_ratings = pd.concat([base_ratings, new_ratings], ignore_index=True)
    return base_ratings

def save_new_ratings(new_ratings_df):
    if os.path.exists(NEW_RATINGS_FILE):
        existing = pd.read_csv(NEW_RATINGS_FILE)
        combined = pd.concat([existing, new_ratings_df], ignore_index=True)
    else:
        combined = new_ratings_df
    combined.to_csv(NEW_RATINGS_FILE, index=False)
