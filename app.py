import streamlit as st
import pandas as pd
import numpy as np
import time
from utils import load_data, load_video_model, get_recommendations, get_user_favorites

# Page Configuration
st.set_page_config(
    page_title="Cinematch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for "Stunning" Aesthetics
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Card Styles */
    .movie-card {
        background-color: #262730;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #363945;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        border-color: #ff4b4b;
    }
    
    .movie-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
        color: #ffffff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .movie-genre {
        font-size: 0.85rem;
        color: #a0a0a0;
        margin-bottom: 12px;
    }
    
    .movie-score {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .score-high { background-color: rgba(76, 175, 80, 0.2); color: #4caf50; }
    .score-med { background-color: rgba(255, 193, 7, 0.2); color: #ffeb3b; }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1c24;
    }

    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #ff4b4b !important;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🎬 Cinematch AI")
st.markdown("### Discover your next favorite movie with the power of Deep Learning.")
st.markdown("---")

# Caching Data Loading
@st.cache_resource
def get_data_and_model():
    df, movie_df, user_encoder, movie_encoder = load_data()
    num_users = len(user_encoder.classes_)
    num_movies = len(movie_encoder.classes_)
    model = load_video_model(num_users, num_movies)
    return df, movie_df, user_encoder, movie_encoder, model

try:
    with st.spinner("Loading AI Model and Database..."):
        df, movie_df, user_encoder, movie_encoder, model = get_data_and_model()
        time.sleep(1) # Fake delay for dramatic effect
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

if model is None:
    st.error("Failed to load the model. Please check the model file path.")
    st.stop()

# Sidebar: User Selection
st.sidebar.header("User Profile")
st.sidebar.markdown("Select a User ID to generate personalized recommendations.")

# Get list of unique users
all_users = sorted(df["userId"].unique())
selected_user = st.sidebar.selectbox("Select User ID", all_users)

# Generate Button
if st.sidebar.button("Generate Recommendations", type="primary"):
    with st.spinner("Analyzing viewing patterns..."):
        
        # 1. Show User's Favorites
        st.subheader(f"❤️ Favorites for User {selected_user}")
        favorites = get_user_favorites(selected_user, df, movie_df)
        
        if favorites:
            cols = st.columns(5)
            for idx, movie in enumerate(favorites):
                with cols[idx % 5]:
                    st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title" title="{movie['title']}">{movie['title']}</div>
                            <div class="movie-genre">{movie['genres'].split('|')[0]}</div>
                            <div class="movie-score score-high">★ {movie['rating']}</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No ratings found for this user.")

        st.markdown("---")
        
        # 2. Show Recommendations
        st.subheader("✨ Top Picks For You")
        
        start_time = time.time()
        recommendations = get_recommendations(
            selected_user, model, movie_df, df, user_encoder, movie_encoder
        )
        prediction_time = time.time() - start_time
        
        st.caption(f"Generated in {prediction_time:.3f} seconds using Neural Collaborative Filtering.")

        if recommendations:
            cols = st.columns(5)
            for idx, movie in enumerate(recommendations):
                # Calculate display score (just for visuals, mapped from 0-1 to percentage or stars)
                # The prediction is sigmoid 0-1.
                score = movie['score']
                score_pct = int(score * 100)
                score_class = "score-high" if score > 0.7 else "score-med"
                
                with cols[idx % 5]:
                    st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title" title="{movie['title']}">{movie['title']}</div>
                            <div class="movie-genre">{movie['genres'].split('|')[0]}</div>
                            <div class="movie-score {score_class}">Match: {score_pct}%</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Could not generate recommendations. User might be new or data missing.")

else:
    # Landing state
    st.info("👈 Select a User ID from the sidebar and click 'Generate Recommendations'")
    
    # Show some stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Users", len(all_users))
    c2.metric("Total Movies", len(movie_df))
    c3.metric("Total Ratings", len(df))

