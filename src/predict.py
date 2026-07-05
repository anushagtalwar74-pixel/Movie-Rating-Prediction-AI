"""
===============================================================================
                            CineRate AI
                     Professional Prediction Engine
-------------------------------------------------------------------------------
Author      : Anusha G Talawar
Project     : Movie Rating Prediction using Machine Learning
Version     : 2.0 Professional
===============================================================================
"""

import logging
import warnings
import joblib
import pandas as pd

from src.config import *

warnings.filterwarnings("ignore")

# =============================================================================
# Logger
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =============================================================================
# Movie Rating Predictor
# =============================================================================

class MovieRatingPredictor:

    """
    Professional prediction class.
    """

    def __init__(self):

        self.model = None

    # -------------------------------------------------------------------------
    # Load Trained Model
    # -------------------------------------------------------------------------

    def load_model(self):

        logger.info("=" * 70)
        logger.info("Loading Trained Model")
        logger.info("=" * 70)

        self.model = joblib.load(MODEL_FILE)

        logger.info("Model Loaded Successfully.")

    # -------------------------------------------------------------------------
    # Prepare Input
    # -------------------------------------------------------------------------

    def prepare_input(
        self,
        genre,
        director,
        actor1,
        actor2,
        actor3,
        year,
        duration,
        votes
    ):

        movie_age = 2026 - int(year)

        genre_count = len(str(genre).split(","))

        is_multi = int(genre_count > 1)

        sample = pd.DataFrame({

            "Genre": [genre],

            "Director": [director],

            "Actor 1": [actor1],

            "Actor 2": [actor2],

            "Actor 3": [actor3],

            "Year": [year],

            "Duration": [duration],

            "Votes": [votes],

            "Movie_Age": [movie_age],

            "Genre_Count": [genre_count],

            "Is_MultiGenre": [is_multi]

        })

        return sample
    
        # -------------------------------------------------------------------------
    # Predict Rating
    # -------------------------------------------------------------------------

    def predict(
        self,
        genre,
        director,
        actor1,
        actor2,
        actor3,
        year,
        duration,
        votes
    ):

        if self.model is None:

            self.load_model()

        sample = self.prepare_input(

            genre,
            director,
            actor1,
            actor2,
            actor3,
            year,
            duration,
            votes

        )

        prediction = float(

            self.model.predict(sample)[0]

        )

        prediction = max(1.0, min(10.0, prediction))

        if prediction >= 8:

            category = "⭐⭐⭐⭐⭐ Excellent"

        elif prediction >= 7:

            category = "⭐⭐⭐⭐ Very Good"

        elif prediction >= 6:

            category = "⭐⭐⭐ Good"

        elif prediction >= 5:

            category = "⭐⭐ Average"

        else:

            category = "⭐ Below Average"

        confidence = "High"

        return {

            "Predicted Rating": round(prediction, 2),

            "Category": category,

            "Confidence": confidence

        }


# =============================================================================
# Test Prediction
# =============================================================================

if __name__ == "__main__":

    predictor = MovieRatingPredictor()

    result = predictor.predict(

        genre="Action, Drama",

        director="S. S. Rajamouli",

        actor1="Prabhas",

        actor2="Rana Daggubati",

        actor3="Anushka Shetty",

        year=2017,

        duration=168,

        votes=250000

    )

    print("\n")

    print("=" * 60)

    print("Prediction Result")

    print("=" * 60)

    for key, value in result.items():

        print(f"{key:20}: {value}")

    print("=" * 60)