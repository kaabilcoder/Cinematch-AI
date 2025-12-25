# 🎬 Cinematch AI - Movie Recommendation System

**Cinematch AI** is a cutting-edge movie recommendation engine built to provide personalized movie suggestions using Deep Learning. By leveraging **Neural Collaborative Filtering (NCF)**, this application analyzes user preferences and viewing history to predict and recommend movies that users are most likely to enjoy.

---

## 🚀 Features

- **Personalized Recommendations**: Generates a tailored list of movies for a selected user based on their historical ratings.
- **Deep Learning Powered**: Utilizes a pre-trained Neural Collaborative Filtering model (Keras) for high-accuracy predictions.
- **Interactive Dashboard**: A visually stunning, dark-themed UI built with Streamlit, featuring hover effects and responsive movie cards.
- **User Insights**: Displays the user's top "Favorite" movies (5-star ratings) to provide context for the recommendations.
- **Efficient Performance**: Optimized data loading and caching to ensure a smooth user experience.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Python-based web framework)
- **Backend/Model**: [TensorFlow/Keras](https://www.tensorflow.org/) (Deep Learning)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Dataset**: [MovieLens Dataset](https://grouplens.org/datasets/movielens/) (Small version included)

---

## 📂 Project Structure

```bash
movieRecommendationSystem/
├── app.py                  # Main Streamlit application entry point
├── utils.py                # Utility functions for data loading, preprocessing, and inference
├── verify_backend.py       # Script to verify model loading and backend logic
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── notebook/               # Jupyter notebooks for EDA and model training
│   └── movie-recommendation-system.ipynb
├── model/                  # Directory containing the trained model
│   └── recommender_model.keras
└── data/                   # Dataset files (MovieLens)
    ├── movies.csv          # Movie metadata (title, genres)
    ├── ratings.csv         # User ratings
    ├── ...
```

---

## 🏁 Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd movieRecommendationSystem
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🖥️ Usage

1.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

2.  **Navigate to the local URL:**
    Usually `http://localhost:8501`.

3.  **Interact with the App:**
    - Use the **Sidebar** to select a **User ID**.
    - Click **"Generate Recommendations"** to see the results.
    - View the user's past favorites and the top predicted movies for them.

---

## 🧠 Model Details

The core of this project is a **Neural Collaborative Filtering (NCF)** model.
- **Architecture**: Embeddings for Users and Movies combined with dense layers to predict a user's likelihood of interacting with a movie.
- **Training**: Trained on the MovieLens dataset, optimizing for a binary classification or regression task (rating prediction).
- **Input**: User ID and Movie ID.
- **Output**: A predicted score (probability) indicating recommendation strength.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Created by [Your Name] for [Course Name/Project Name].*
