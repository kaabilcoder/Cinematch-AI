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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Global Styles */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 15, 20) 0%, rgb(5, 5, 5) 90.2%);
        color: #fafafa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.03em;
        text-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(20, 20, 30, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Card Styles - Glassmorphism */
    .movie-card {
        background: rgba(40, 40, 50, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0) 100%);
        pointer-events: none;
    }
    
    .movie-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        border-color: rgba(255, 75, 75, 0.5);
    }
    
    .movie-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 10px;
        color: #ffffff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    .movie-genre {
        font-size: 0.9rem;
        color: #b0b0b0;
        margin-bottom: 16px;
        display: inline-block;
        background: rgba(255,255,255,0.05);
        padding: 4px 10px;
        border-radius: 20px;
    }
    
    .movie-score {
        position: absolute;
        bottom: 20px;
        right: 20px;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    
    .score-high { 
        background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%); 
        color: white; 
    }
    .score-med { 
        background: linear-gradient(135deg, #ffc107 0%, #ff8f00 100%); 
        color: black; 
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(10, 10, 10, 0.9);
        color: #888;
        text-align: center;
        padding: 15px 0;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        z-index: 100;
        backdrop-filter: blur(5px);
    }
    
    .footer a {
        color: #ff4b4b;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.3s;
    }
    
    .footer a:hover {
        color: #ff8f8f;
        text-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
    }

    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #ff4b4b !important;
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #ff4b4b 0%, #cc0000 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.5);
    }
    
    /* Decoration on top */
    .top-decoration {
        height: 4px;
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b, #ff8f00, #4caf50);
        position: fixed;
        top: 0;
        left: 0;
        z-index: 9999;
    }
</style>
<div class="top-decoration"></div>
""", unsafe_allow_html=True)

# Application Header
st.title("🎬 Cinematch AI")
st.markdown("<h3 style='opacity: 0.8; font-weight: 400;'>Discover your next favorite movie with the power of Deep Learning.</h3>", unsafe_allow_html=True)
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

# Footer
st.markdown("""
<div class="footer">
    <p>Made with ❤️ by <b>Saurabh Kumar Sahu</b> | <a href="https://github.com/SaurabhKumarSahu" target="_blank">Visit my GitHub</a></p>
</div>
""", unsafe_allow_html=True)

