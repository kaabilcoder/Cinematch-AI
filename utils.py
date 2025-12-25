import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Define the RecommenderNet class (Same as in the notebook)
@keras.utils.register_keras_serializable()
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim

        self.user_embedding = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )

        self.movie_embedding = layers.Embedding(
            input_dim=num_movies,
            output_dim=embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )

        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_idx = inputs[:, 0]
        movie_idx = inputs[:, 1]

        user_vec = self.user_embedding(user_idx)
        movie_vec = self.movie_embedding(movie_idx)

        user_bias = self.user_bias(user_idx)
        movie_bias = self.movie_bias(movie_idx)

        # Dot product between user and movie embeddings
        dot = tf.reduce_sum(user_vec * movie_vec, axis=1, keepdims=True)

        x = dot + user_bias + movie_bias

        # Output normalized rating in [0,1]
        return tf.nn.sigmoid(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.num_users,
            "num_movies": self.num_movies,
            "embedding_dim": self.embedding_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def load_data():
    """Loads datasets and prepares encoders."""
    data_dir = Path("data/ml-latest-small")
    # If data is not in data/ml-latest-small (based on notebook logic it downloaded there), 
    # but user said @[model] is in /model, and @[notebook] in /notebook.
    # The list_dir output showed 'data' folder in root.
    # Inside data directory: README.txt, links.csv, movies.csv, ratings.csv, tags.csv
    # So we should read from "data/".
    
    ratings_file = Path("data/ratings.csv")
    movies_file = Path("data/movies.csv")

    df = pd.read_csv(ratings_file)
    movie_df = pd.read_csv(movies_file)

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    df["user_idx"] = user_encoder.fit_transform(df["userId"])
    df["movie_idx"] = movie_encoder.fit_transform(df["movieId"])

    return df, movie_df, user_encoder, movie_encoder

def load_video_model(num_users, num_movies):
    """Loads the trained model."""
    model_path = "model/recommender_model.keras"
    try:
        # Instantiate model with correct parameters
        model = RecommenderNet(num_users, num_movies)
        # Build the model to create weights variables
        model.build(input_shape=(None, 2))
        # Load weights
        model.load_weights(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_recommendations(user_id, model, movie_df, df, user_encoder, movie_encoder, top_n=10):
    """Generates movie recommendations for a specific user."""
    
    # Check if user exists
    if user_id not in df["userId"].values:
        return []

    # Get movies watched by user
    movies_watched_by_user = df[df["userId"] == user_id]
    
    # Get movies NOT watched
    all_movie_ids = movie_df["movieId"].values
    movies_not_watched = np.setdiff1d(all_movie_ids, movies_watched_by_user["movieId"].values)

    # Filter movies that are not in the encoder (new movies not seen during training)
    # The model can only predict for movies it has seen (or we need to handle unseen).
    # Since it's collaborative filtering with embeddings, we are limited to known movies.
    known_movie_ids = set(movie_encoder.classes_)
    movies_not_watched = [m for m in movies_not_watched if m in known_movie_ids]

    if not movies_not_watched:
        return []

    # Prepare input for model
    user_encoder_idx = user_encoder.transform([user_id])[0]
    movie_encoded_indices = movie_encoder.transform(movies_not_watched)
    
    user_movie_array = np.column_stack(
        ([user_encoder_idx] * len(movie_encoded_indices), movie_encoded_indices)
    )

    # Predict
    # predict returns normalized ratings [0, 1]
    ratings = model.predict(user_movie_array, verbose=0).flatten()

    # Get top N indices
    top_indices_in_not_watched = ratings.argsort()[-top_n:][::-1]
    recommended_movie_ids = [movies_not_watched[i] for i in top_indices_in_not_watched]
    
    # Get details
    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
    
    # Sort them by the prediction score order (because 'isin' doesn't preserve order)
    # Create a mapping from ID to rating for sorting
    id_to_rating = {m_id: r for m_id, r in zip(movies_not_watched, ratings)}
    
    recommended_movies_list = []
    for m_id in recommended_movie_ids:
        row = recommended_movies[recommended_movies["movieId"] == m_id].iloc[0]
        recommended_movies_list.append({
            "title": row["title"],
            "genres": row["genres"],
            "score": id_to_rating[m_id]
        })
        
    return recommended_movies_list

def get_user_favorites(user_id, df, movie_df, top_n=5):
    """Returns the top rated movies by the user."""
    user_data = df[df["userId"] == user_id].sort_values(by="rating", ascending=False).head(top_n)
    
    favorites = []
    for _, row in user_data.iterrows():
        movie_row = movie_df[movie_df["movieId"] == row["movieId"]]
        if not movie_row.empty:
            favorites.append({
                "title": movie_row.iloc[0]["title"],
                "genres": movie_row.iloc[0]["genres"],
                "rating": row["rating"]
            })
    return favorites
