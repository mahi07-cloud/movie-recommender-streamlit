import streamlit as st
import pandas as pd

from recommender.data_loader import load_movies, load_ratings
from recommender.model_trainer import train_model
from recommender.recommend import (
    build_content_model,
    get_content_similar_movies,
    get_hybrid_recommendations,
    save_user_ratings,
    load_user_ratings
)

# Load data
movies = load_movies()
ratings = load_ratings()
model = train_model(ratings)

# Content-based model
cosine_sim = build_content_model(movies)

# Session State
if "user_id" not in st.session_state:
    st.session_state.user_id = 9999
if "new_ratings" not in st.session_state:
    st.session_state.new_ratings = []

# App UI
st.title("🎥 Movie Recommender System")
menu = st.sidebar.radio("Choose Action", ["Recommend for Me", "Find Similar Movies", "Rate Movies"])

if menu == "Recommend for Me":
    st.header("📽 Personalized Movie Recommendations")
    user_id = st.number_input("Enter your user ID", min_value=1, value=1)
    recs = get_hybrid_recommendations(user_id, model, movies, ratings, cosine_sim)
    st.write("🎯 Top Recommendations:")
    st.dataframe(recs[['title', 'genres']])

elif menu == "Find Similar Movies":
    st.header("🎞 Content-Based: Similar Movies")
    movie_title = st.selectbox("Pick a movie", movies['title'].sort_values().unique())
    if movie_title:
        similar = get_content_similar_movies(movie_title, movies, cosine_sim)
        st.write("🔍 Similar Movies:")
        st.dataframe(similar[['title', 'genres']])

elif menu == "Rate Movies":
    st.header("⭐ Add Your Ratings")
    user_id = st.number_input("New User ID", min_value=1000, value=st.session_state.user_id)
    movie_title = st.selectbox("Select Movie", movies['title'].sort_values().unique(), key="rate_movie")
    rating = st.slider("Your Rating", 0.5, 5.0, 3.0, 0.5)
    
    if st.button("Submit Rating"):
        movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
        st.session_state.new_ratings.append({"userId": user_id, "movieId": movie_id, "rating": rating})
        st.success(f"Rating saved for '{movie_title}'")
    
    if st.button("Save All Ratings to File"):
        if st.session_state.new_ratings:
            new_df = pd.DataFrame(st.session_state.new_ratings)
            save_user_ratings(new_df)
            st.success("✅ Ratings saved to file!")
            st.session_state.new_ratings.clear()
        else:
            st.warning("No new ratings to save.")
