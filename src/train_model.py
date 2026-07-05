"""
===============================================================================
                            CineRate AI
                    Professional Model Training Pipeline
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
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor

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
# Movie Rating Trainer
# =============================================================================

class MovieRatingTrainer:

    """
    Professional Movie Rating Trainer
    """

    def __init__(self):

        self.df = None

        self.X = None

        self.y = None

        self.X_train = None

        self.X_test = None

        self.y_train = None

        self.y_test = None

        self.preprocessor = None

        self.models = {}

        self.results = {}

        self.best_model = None

        self.best_model_name = ""

    # -------------------------------------------------------------------------
    # Load Processed Dataset
    # -------------------------------------------------------------------------

    def load_dataset(self):

        logger.info("=" * 70)
        logger.info("Loading Processed Dataset")
        logger.info("=" * 70)

        self.df = pd.read_csv(PROCESSED_DATA)

        logger.info(f"Dataset Shape : {self.df.shape}")

    # -------------------------------------------------------------------------
    # Prepare Features
    # -------------------------------------------------------------------------

    def prepare_features(self):

        logger.info("=" * 70)
        logger.info("Preparing Features")
        logger.info("=" * 70)

        feature_columns = [

            "Genre",

            "Director",

            "Actor 1",

            "Actor 2",

            "Actor 3",

            "Year",

            "Duration",

            "Votes",

            "Movie_Age",

            "Genre_Count",

            "Is_MultiGenre"

        ]

        self.X = self.df[feature_columns]

        self.y = self.df[TARGET_COLUMN]

        logger.info(f"Total Features : {len(feature_columns)}")

        logger.info(f"Training Samples : {len(self.X)}")

    # -------------------------------------------------------------------------
    # Split Dataset
    # -------------------------------------------------------------------------

    def split_dataset(self):

        logger.info("=" * 70)
        logger.info("Splitting Dataset")
        logger.info("=" * 70)

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test

        ) = train_test_split(

            self.X,

            self.y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE

        )

        logger.info(f"Train Shape : {self.X_train.shape}")

        logger.info(f"Test Shape : {self.X_test.shape}")

    # -------------------------------------------------------------------------
    # Build Preprocessor
    # -------------------------------------------------------------------------

    def build_preprocessor(self):

        logger.info("=" * 70)
        logger.info("Building Preprocessor")
        logger.info("=" * 70)

        categorical = [

            "Genre",

            "Director",

            "Actor 1",

            "Actor 2",

            "Actor 3"

        ]

        numeric = [

            "Year",

            "Duration",

            "Votes",

            "Movie_Age",

            "Genre_Count",

            "Is_MultiGenre"

        ]

        self.preprocessor = ColumnTransformer(

            transformers=[

                (

                    "categorical",

                    Pipeline(

                        [

                            (

                                "imputer",

                                SimpleImputer(

                                    strategy="most_frequent"

                                )

                            ),

                            (

                                "encoder",

                                OneHotEncoder(

                                    handle_unknown="ignore"

                                )

                            )

                        ]

                    ),

                    categorical

                ),

                (

                    "numeric",

                    Pipeline(

                        [

                            (

                                "imputer",

                                SimpleImputer(

                                    strategy="median"

                                )

                            )

                        ]

                    ),

                    numeric

                )

            ]

        )

        logger.info("Preprocessor Ready.")
        
        # -------------------------------------------------------------------------
    # Build Machine Learning Models
    # -------------------------------------------------------------------------

    def build_models(self):

        logger.info("=" * 70)
        logger.info("Building Machine Learning Models")
        logger.info("=" * 70)

        self.models = {

            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(

                n_estimators=300,

                max_depth=20,

                random_state=RANDOM_STATE,

                n_jobs=-1

            ),

            "Gradient Boosting": GradientBoostingRegressor(

                n_estimators=250,

                learning_rate=0.05,

                max_depth=4,

                random_state=RANDOM_STATE

            ),

            "XGBoost": XGBRegressor(

                n_estimators=300,

                learning_rate=0.05,

                max_depth=6,

                subsample=0.8,

                colsample_bytree=0.8,

                random_state=RANDOM_STATE,

                objective="reg:squarederror"

            )

        }

        logger.info(f"Total Models : {len(self.models)}")

    # -------------------------------------------------------------------------
    # Train All Models
    # -------------------------------------------------------------------------

    def train_models(self):

        logger.info("=" * 70)
        logger.info("Training Models")
        logger.info("=" * 70)

        best_score = -999

        for model_name, model in self.models.items():

            logger.info(f"Training {model_name}...")

            pipeline = Pipeline(

                [

                    ("preprocessor", self.preprocessor),

                    ("model", model)

                ]

            )

            pipeline.fit(

                self.X_train,

                self.y_train

            )

            predictions = pipeline.predict(

                self.X_test

            )

            r2 = r2_score(

                self.y_test,

                predictions

            )

            mae = mean_absolute_error(

                self.y_test,

                predictions

            )

            rmse = root_mean_squared_error(

                self.y_test,

                predictions

            )

            self.results[model_name] = {

                "Pipeline": pipeline,

                "R2": r2,

                "MAE": mae,

                "RMSE": rmse

            }

            logger.info(

                f"{model_name} | "
                f"R²={r2:.4f} | "
                f"MAE={mae:.4f} | "
                f"RMSE={rmse:.4f}"

            )

            if r2 > best_score:

                best_score = r2

                self.best_model = pipeline

                self.best_model_name = model_name

        logger.info("=" * 70)
        logger.info(f"Best Model : {self.best_model_name}")
        logger.info("=" * 70)
        
        # -------------------------------------------------------------------------
    # Display Model Comparison
    # -------------------------------------------------------------------------

    def model_report(self):

        logger.info("=" * 70)
        logger.info("Model Performance Report")
        logger.info("=" * 70)

        report = pd.DataFrame({

            "Model": self.results.keys(),

            "R2 Score": [

                round(v["R2"], 4)

                for v in self.results.values()

            ],

            "MAE": [

                round(v["MAE"], 4)

                for v in self.results.values()

            ],

            "RMSE": [

                round(v["RMSE"], 4)

                for v in self.results.values()

            ]

        })

        report = report.sort_values(

            by="R2 Score",

            ascending=False

        )

        print("\n")

        print(report)

        print("\n")

    # -------------------------------------------------------------------------
    # Save Best Model
    # -------------------------------------------------------------------------

    def save_model(self):

        logger.info("=" * 70)
        logger.info("Saving Model")
        logger.info("=" * 70)

        MODEL_DIR.mkdir(

            parents=True,

            exist_ok=True

        )

        joblib.dump(

            self.best_model,

            MODEL_FILE

        )

        logger.info(

            f"Model Saved : {MODEL_FILE}"

        )

    # -------------------------------------------------------------------------
    # Save Preprocessor
    # -------------------------------------------------------------------------

    def save_preprocessor(self):

        logger.info("=" * 70)
        logger.info("Saving Preprocessor")
        logger.info("=" * 70)

        joblib.dump(

            self.preprocessor,

            PREPROCESSOR_FILE

        )

        logger.info(

            f"Preprocessor Saved : {PREPROCESSOR_FILE}"

        )

    # -------------------------------------------------------------------------
    # Complete Pipeline
    # -------------------------------------------------------------------------

    def run(self):

        logger.info("=" * 70)
        logger.info("Movie Rating Prediction Training Started")
        logger.info("=" * 70)

        self.load_dataset()

        self.prepare_features()

        self.split_dataset()

        self.build_preprocessor()

        self.build_models()

        self.train_models()

        self.model_report()

        self.save_model()

        self.save_preprocessor()

        logger.info("=" * 70)
        logger.info("Training Completed Successfully")
        logger.info("=" * 70)

        logger.info(f"Best Model : {self.best_model_name}")

        return self.best_model
    
# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    trainer = MovieRatingTrainer()

    trainer.run()