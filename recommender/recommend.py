import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD
import os

DATA_PATH = "data"
RATINGS_FILE = os.path.join(DATA_PATH, "ratings.csv")
MOVIES_FILE = os.path.join(DATA_PATH, "movies.csv")

# Content-based filtering setup
def build_content_model(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Get similar movies
def get_content_similar_movies(movie_title, movies, cosine_sim):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

# Hybrid: content + collaborative
def get_hybrid_recommendations(user_id, model, movies, ratings, cosine_sim):
    user_rated = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    scores = []
    for idx, row in movies.iterrows():
        mid = row['movieId']
        if mid not in user_rated:
            try:
                cf_score = model.predict(user_id, mid).est
                content_idx = movies[movies['movieId'] == mid].index[0]
                content_score = cosine_sim[content_idx].mean()
                final_score = 0.7 * cf_score + 0.3 * content_score
                scores.append((mid, final_score))
            except:
                continue
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
    rec_movies = movies[movies['movieId'].isin([x[0] for x in scores])]
    return rec_movies

# Save new ratings
def save_user_ratings(new_ratings, ratings_file=RATINGS_FILE):
    if os.path.exists(ratings_file):
        ratings_df = pd.read_csv(ratings_file)
        updated_df = pd.concat([ratings_df, new_ratings], ignore_index=True)
    else:
        updated_df = new_ratings
    updated_df.to_csv(ratings_file, index=False)

# Load user ratings
def load_user_ratings():
    if os.path.exists(RATINGS_FILE):
        return pd.read_csv(RATINGS_FILE)
    else:
        return pd.DataFrame(columns=["userId", "movieId", "rating"])
