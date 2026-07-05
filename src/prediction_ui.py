"""
===============================================================================
                        CineRate AI Prediction UI
===============================================================================
"""

import streamlit as st

from src.predict import MovieRatingPredictor
from src.helper import (
    rating_badge,
    rating_color,
    confidence_level
)


class PredictionUI:

    def __init__(self):

        self.predictor = MovieRatingPredictor()

    def render(self):

        st.markdown("## 🤖 AI Movie Rating Prediction")

        left, right = st.columns([1.3, 1])

        with left:

            with st.form("prediction_form"):

                genre = st.text_input(
                    "🎭 Genre",
                    placeholder="Drama, Action"
                )

                director = st.text_input(
                    "🎬 Director",
                    placeholder="Rajkumar Hirani"
                )

                actor1 = st.text_input(
                    "⭐ Lead Actor",
                    placeholder="Aamir Khan"
                )

                actor2 = st.text_input(
                    "⭐ Supporting Actor",
                    placeholder="Sharman Joshi"
                )

                actor3 = st.text_input(
                    "⭐ Supporting Actress",
                    placeholder="Kareena Kapoor"
                )

                col1, col2 = st.columns(2)

                with col1:

                    year = st.slider(
                        "📅 Release Year",
                        1950,
                        2026,
                        2020
                    )

                    duration = st.slider(
                        "⏱ Duration (Minutes)",
                        60,
                        240,
                        140
                    )

                with col2:

                    votes = st.number_input(
                        "👍 IMDb Votes",
                        min_value=0,
                        value=5000,
                        step=100
                    )

                predict = st.form_submit_button(
                    "🚀 Predict Rating",
                    use_container_width=True
                )

        with right:

            st.markdown("### 🎯 Prediction Result")

            output = st.empty()

        if predict:

            if genre.strip() == "":

                st.warning("Please enter Genre.")

                return

            result = self.predictor.predict(

                genre=genre,

                director=director,

                actor1=actor1,

                actor2=actor2,

                actor3=actor3,

                year=year,

                duration=duration,

                votes=votes

            )

            rating = result["Predicted Rating"]

            color = rating_color(rating)

            badge = rating_badge(rating)

            confidence = confidence_level(rating)

            output.markdown(

                f"""
<div class="prediction-card">

<h1 style="color:{color};font-size:65px;">

⭐ {rating}

</h1>

<h3>{badge}</h3>

<hr>

<h4>Confidence</h4>

<h2>{confidence}</h2>

</div>
""",

                unsafe_allow_html=True

            )