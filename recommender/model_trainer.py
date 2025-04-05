import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import os

# Train the collaborative filtering model
def train_model(ratings_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    return model
