from utils import load_data, load_video_model, get_recommendations
import numpy as np

print("Loading data...")
try:
    df, movie_df, user_encoder, movie_encoder = load_data()
    print("Data loaded successfully.")
except Exception as e:
    print(f"Data loading failed: {e}")
    exit(1)

print("\nLoading model...")
try:
    num_users = len(user_encoder.classes_)
    num_movies = len(movie_encoder.classes_)
    model = load_video_model(num_users, num_movies)
    if model:
        print("Model loaded successfully.")
    else:
        print("Model loading returned None.")
        exit(1)
except Exception as e:
    print(f"Model loading failed: {e}")
    exit(1)

print("\nTesting recommendations...")
try:
    # Pick a valid user ID. The notebook showed user_id 341 as example.
    # Let's pick a random one from df
    if not df.empty:
        test_user_id = df["userId"].iloc[0]
        print(f"Testing for user ID: {test_user_id}")
        recs = get_recommendations(test_user_id, model, movie_df, df, user_encoder, movie_encoder)
        print(f"Got {len(recs)} recommendations.")
        if recs:
            print("Top recommendation:", recs[0])
    else:
        print("Dataset is empty.")
except Exception as e:
    print(f"Recommendation failed: {e}")
    exit(1)
