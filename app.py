"""
===============================================================================
                                CineRate AI
               Professional Movie Rating Prediction Platform
===============================================================================
Author  : Anusha G Talawar
Version : 3.0 Professional
===============================================================================
"""

import json
from pathlib import Path
import base64
import streamlit as st
from streamlit_lottie import st_lottie

from src.components import UIComponents
from src.predict import MovieRatingPredictor
from src.visualization import MovieVisualizer

BASE_DIR = Path(__file__).parent
ASSETS = BASE_DIR / "assets"
# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="🎬 CineRate AI",
    page_icon=str(ASSETS / "favicon.ico"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# LOAD CSS
# =============================================================================

BASE_DIR = Path(__file__).parent

CSS_FILE = BASE_DIR / "style.css"

with open(CSS_FILE, encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )
# =============================================================================
# BACKGROUND FUNCTION
# =============================================================================

def set_background(image_path):
    try:
        with open(image_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background: linear-gradient(
                    rgba(10,10,10,0.85),
                    rgba(10,10,10,0.85)
                ),
                url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

# <-- CALL IT HERE
set_background(ASSETS / "bg.jpg")

# =============================================================================
# LOAD LOTTIE
# =============================================================================

def load_lottie(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None




# =============================================================================
# LOAD MODEL
# =============================================================================

predictor = MovieRatingPredictor()
visualizer = MovieVisualizer()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    if Path(ASSETS / "logo.png").exists():
        
        try:
            st.image(ASSETS / "logo.png", width=180)
        except Exception:
            pass

    st.title("🎬 CineRate AI")

    st.caption("Professional Movie Rating Prediction")

    st.divider()

    st.success("✅ Model Loaded")

    st.info("🏆 Best Model : XGBoost")

    st.metric("Movies", "7,919")

    st.metric("Algorithms", "4")

    st.metric("Best R²", "0.396")

    st.divider()

    st.caption("Version 3.0 Professional")

# =============================================================================
# HERO SECTION
# =============================================================================

left, right = st.columns([1.5, 1])

with left:

    if Path(ASSETS / "logo.pn").exists():
        st.image(ASSETS / "logo.png", width=170)

    UIComponents.hero()



st.divider()

from pathlib import Path

col1, col2, col3 = st.columns([1,1,1])

with col2:
    if Path(ASSETS / "movie_icon.png").exists():
        st.image(ASSETS / "movie_icon.png", width=150)
# =============================================================================
# KPI SECTION
# =============================================================================

stats = visualizer.dashboard_stats()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("🎥 Movies", stats["Total Movies"])

with c2:
    st.metric("⭐ Average Rating", stats["Average Rating"])

with c3:
    st.metric("🎬 Directors", stats["Total Directors"])

with c4:
    st.metric("👥 Actors", stats["Total Actors"])

st.divider()
# =============================================================================
# AI MOVIE RATING PREDICTION
# =============================================================================

UIComponents.section("AI Movie Rating Prediction", "🤖")

left, right = st.columns([1.2, 1])

with left:

    movie_name = st.text_input(
        "🎬 Movie Name",
        placeholder="Enter movie name..."
    )

    genre = st.text_input(
        "🎭 Genre",
        placeholder="Drama, Action..."
    )

    director = st.text_input(
        "🎬 Director",
        placeholder="Director Name"
    )

    actor1 = st.text_input(
        "⭐ Lead Actor",
        placeholder="Actor Name"
    )

    actor2 = st.text_input(
        "🎭 Supporting Actor",
        placeholder="Actor Name"
    )

    actor3 = st.text_input(
        "🎭 Supporting Actress",
        placeholder="Actor Name"
    )

    year = st.number_input(
        "📅 Release Year",
        min_value=1950,
        max_value=2035,
        value=2024
    )

    duration = st.slider(
        "⏱ Duration (Minutes)",
        60,
        240,
        130
    )

    votes = st.number_input(
        "👍 Expected Votes",
        min_value=0,
        value=2000
    )

    predict_btn = st.button(
        "🚀 Predict IMDb Rating",
        width="stretch"
    )

with right:

    st.info(
        """
### 🤖 AI Prediction Engine

This prediction uses the trained **XGBoost Regressor**
developed during model training.

Features considered:

✔ Genre

✔ Director

✔ Main Actors

✔ Release Year

✔ Duration

✔ Votes

The model predicts an estimated IMDb rating.
"""
    )

# =============================================================================
# PREDICTION
# =============================================================================

if predict_btn:

    try:

        result = predictor.predict(
            genre=genre,
            director=director,
            actor1=actor1,
            actor2=actor2,
            actor3=actor3,
            year=year,
            duration=duration,
            votes=votes,
        )

        st.success("Prediction Completed Successfully")

        score = result["Predicted Rating"]
        category = result["Category"]
        confidence = result["Confidence"]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "⭐ Predicted Rating",
                f"{score:.2f}/10"
            )

        with c2:
            st.metric(
                "Category",
                category
            )

        with c3:
            st.metric(
                "Confidence",
                confidence
            )

        st.progress(score / 10)

        st.balloons()

    except Exception as e:

        st.error(f"Prediction Failed: {e}")

st.divider()
# =============================================================================
# ANALYTICS DASHBOARD
# =============================================================================

UIComponents.section("Movie Analytics Dashboard", "📊")

tab1, tab2, tab3 = st.tabs(
    [
        "📈 Overview",
        "🎬 Movies",
        "🎭 Industry"
    ]
)

# =============================================================================
# TAB 1 — OVERVIEW
# =============================================================================

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            visualizer.rating_distribution(),
            width="stretch"
        )

    with col2:
        st.plotly_chart(
            visualizer.genre_distribution(),
            width="stretch"
        )

    st.plotly_chart(
        visualizer.yearly_releases(),
        width="stretch"
    )

# =============================================================================
# TAB 2 — MOVIES
# =============================================================================

with tab2:

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            visualizer.top_movies(),
            width="stretch"
        )

    with col2:

        st.plotly_chart(
            visualizer.votes_vs_rating(),
            width="stretch"
        )

    st.markdown("### 🏆 Highest Rated Movies")

    st.dataframe(
        visualizer.top_movies_table(),
        width="stretch",
        hide_index=True
    )

# =============================================================================
# TAB 3 — INDUSTRY
# =============================================================================

with tab3:

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            visualizer.top_directors(),
            width="stretch"
        )

    with col2:

        st.plotly_chart(
            visualizer.top_actors(),
            width="stretch"
        )

    st.plotly_chart(
        visualizer.correlation_heatmap(),
        width="stretch"
    )

st.divider()
# =============================================================================
# MODEL PERFORMANCE
# =============================================================================

UIComponents.section("Machine Learning Performance", "🏆")

performance = visualizer.model_performance()

st.dataframe(
    performance,
    width="stretch",
    hide_index=True,
)

st.plotly_chart(
    visualizer.model_comparison(),
    width="stretch"
)

st.divider()

# =============================================================================
# PROJECT HIGHLIGHTS
# =============================================================================

UIComponents.section("Project Highlights", "🚀")

c1, c2, c3 = st.columns(3)

with c1:

    st.success(
        """
### 🤖 AI Powered

✔ XGBoost

✔ Random Forest

✔ Gradient Boosting

✔ Linear Regression
"""
    )

with c2:

    st.info(
        """
### 📊 Analytics

✔ Interactive Dashboard

✔ Plotly Charts

✔ Movie Insights

✔ Genre Analysis
"""
    )

with c3:

    st.warning(
        """
### ⚡ Features

✔ Prediction Engine

✔ Modern UI

✔ Responsive Layout

✔ Fast Processing
"""
    )

st.divider()

# =============================================================================
# DATASET OVERVIEW
# =============================================================================

UIComponents.section("Dataset Overview", "📁")

left, right = st.columns([1, 2])

with left:

    st.metric("Movies", "7,919")

    st.metric("Features", "13")

    st.metric("Algorithms", "4")

    st.metric("Best Model", "XGBoost")

with right:

    st.info(
        """
### IMDb India Movies Dataset

The project is trained on the IMDb India Movies Dataset.

Features include:

• Movie Name

• Genre

• Director

• Actors

• Duration

• Release Year

• Votes

• IMDb Rating

After preprocessing, feature engineering and cleaning,
the dataset was used to train multiple Machine Learning
models to predict movie ratings.
"""
    )

st.divider()
# =============================================================================
# ABOUT PROJECT
# =============================================================================

UIComponents.section("About CineRate AI", "🎬")

st.markdown("""
### 🌟 CineRate AI

CineRate AI is a professional Machine Learning application that predicts IMDb
movie ratings using historical movie data.

The project demonstrates a complete end-to-end Data Science workflow:

- 📥 Data Collection & Cleaning
- 🧹 Data Preprocessing
- 📊 Exploratory Data Analysis (EDA)
- ⚙ Feature Engineering
- 🤖 Machine Learning Model Training
- 🏆 Model Evaluation
- 🎨 Interactive Dashboard Development
- 🚀 Streamlit Deployment

This project was built to showcase practical Data Science and Machine Learning
skills in a professional portfolio.
""")

st.divider()

# =============================================================================
# TECHNOLOGY STACK
# =============================================================================

UIComponents.section("Technology Stack", "🛠")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
### 🐍 Backend

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
""")

with col2:
    st.success("""
### 📊 Visualization

- Plotly
- Matplotlib
- Streamlit
- Streamlit Lottie
""")

with col3:
    st.warning("""
### ⚡ Deployment

- Streamlit
- GitHub
- Kaggle Dataset
- Pickle Models
""")

st.divider()

# =============================================================================
# FOOTER
# =============================================================================

UIComponents.footer()

st.markdown(
    """
<div style='text-align:center;padding:20px;color:gray;font-size:15px;'>

⭐ If you like this project, don't forget to give it a ⭐ on GitHub.

<br><br>

Made with ❤️ using Python, Streamlit and Machine Learning

</div>
""",
    unsafe_allow_html=True,
)