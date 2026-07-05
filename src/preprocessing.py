"""
===============================================================================
                            CineRate AI
                 Professional Data Preprocessing Pipeline
-------------------------------------------------------------------------------
Author      : Anusha G Talawar
Project     : Movie Rating Prediction using Machine Learning
Version     : 2.0 Professional
===============================================================================
"""

import logging
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import *

warnings.filterwarnings("ignore")

# =============================================================================
# Configure Logger
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =============================================================================
# Movie Preprocessor
# =============================================================================

class MoviePreprocessor:

    """
    Professional preprocessing pipeline
    """

    REQUIRED_COLUMNS = [
        "Name",
        "Year",
        "Duration",
        "Genre",
        "Rating",
        "Votes",
        "Director",
        "Actor 1",
        "Actor 2",
        "Actor 3"
    ]

    def __init__(self):

        self.df = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------------------------

    def load_dataset(self):

        logger.info("=" * 70)
        logger.info("Loading IMDb Dataset")
        logger.info("=" * 70)

        encodings = [
            "utf-8",
            "latin1",
            "cp1252"
        ]

        for encoding in encodings:

            try:

                self.df = pd.read_csv(
                    RAW_DATA,
                    encoding=encoding
                )

                logger.info(f"Loaded using {encoding}")

                break

            except UnicodeDecodeError:

                continue

        if self.df.empty:

            raise ValueError(
                "Dataset could not be loaded."
            )

        logger.info(f"Rows : {self.df.shape[0]}")
        logger.info(f"Columns : {self.df.shape[1]}")

        return self.df

    # -------------------------------------------------------------------------
    # Validate Dataset
    # -------------------------------------------------------------------------

    def validate_dataset(self):

        logger.info("Validating Dataset...")

        missing = []

        for column in self.REQUIRED_COLUMNS:

            if column not in self.df.columns:

                missing.append(column)

        if missing:

            raise Exception(
                f"Missing Columns : {missing}"
            )

        logger.info("Dataset Validation Successful")

    # -------------------------------------------------------------------------
    # Basic Information
    # -------------------------------------------------------------------------

    def dataset_information(self):

        logger.info("=" * 70)
        logger.info("Dataset Information")
        logger.info("=" * 70)

        logger.info(f"Shape : {self.df.shape}")

        logger.info(
            f"Duplicate Rows : {self.df.duplicated().sum()}"
        )

        logger.info(
            f"Missing Values :\n{self.df.isna().sum()}"
        )

    # -------------------------------------------------------------------------
    # Remove Duplicate Records
    # -------------------------------------------------------------------------

    def remove_duplicates(self):

        before = len(self.df)

        self.df.drop_duplicates(inplace=True)

        after = len(self.df)

        logger.info(
            f"Removed {before-after} duplicate rows."
        )
    
        # -------------------------------------------------------------------------
    # Handle Missing Values
    # -------------------------------------------------------------------------

    def handle_missing_values(self):

        logger.info("=" * 70)
        logger.info("Handling Missing Values")
        logger.info("=" * 70)

        text_columns = [
            "Genre",
            "Director",
            "Actor 1",
            "Actor 2",
            "Actor 3"
        ]

        for column in text_columns:

            self.df[column] = (
                self.df[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

        self.df["Duration"] = self.df["Duration"].fillna("120 min")
        self.df["Votes"] = self.df["Votes"].fillna("0")

        before = len(self.df)

        self.df.dropna(subset=["Rating"], inplace=True)

        removed = before - len(self.df)

        logger.info(f"Removed {removed} rows with missing Rating.")

    # -------------------------------------------------------------------------
    # Clean Year
    # -------------------------------------------------------------------------

    def clean_year(self):

        logger.info("Cleaning Year Column...")

        self.df["Year"] = (
            self.df["Year"]
            .astype(str)
            .str.extract(r"(\d{4})")[0]
        )

        self.df["Year"] = pd.to_numeric(
            self.df["Year"],
            errors="coerce"
        )

        median_year = int(self.df["Year"].median())

        self.df["Year"] = self.df["Year"].fillna(median_year)

        self.df["Year"] = self.df["Year"].astype(int)

    # -------------------------------------------------------------------------
    # Clean Duration
    # -------------------------------------------------------------------------

    def clean_duration(self):

        logger.info("Cleaning Duration Column...")

        self.df["Duration"] = (
            self.df["Duration"]
            .astype(str)
            .str.extract(r"(\d+)")[0]
        )

        self.df["Duration"] = pd.to_numeric(
            self.df["Duration"],
            errors="coerce"
        )

        median_duration = int(self.df["Duration"].median())

        self.df["Duration"] = self.df["Duration"].fillna(median_duration)

        self.df["Duration"] = self.df["Duration"].astype(int)

    # -------------------------------------------------------------------------
    # Clean Votes
    # -------------------------------------------------------------------------

    def clean_votes(self):

        logger.info("Cleaning Votes Column...")

        self.df["Votes"] = (
            self.df["Votes"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        self.df["Votes"] = pd.to_numeric(
            self.df["Votes"],
            errors="coerce"
        )

        median_votes = int(self.df["Votes"].median())

        self.df["Votes"] = self.df["Votes"].fillna(median_votes)

        self.df["Votes"] = self.df["Votes"].astype(int)

    # -------------------------------------------------------------------------
    # Clean Rating
    # -------------------------------------------------------------------------

    def clean_rating(self):

        logger.info("Cleaning Rating Column...")

        self.df["Rating"] = pd.to_numeric(
            self.df["Rating"],
            errors="coerce"
        )

        self.df = self.df[
            self.df["Rating"].between(1, 10)
        ]

    # -------------------------------------------------------------------------
    # Standardize Text Columns
    # -------------------------------------------------------------------------

    def standardize_text_columns(self):

        logger.info("Standardizing Text Columns...")

        columns = [
            "Genre",
            "Director",
            "Actor 1",
            "Actor 2",
            "Actor 3"
        ]

        for column in columns:

            self.df[column] = (
                self.df[column]
                .astype(str)
                .str.strip()
                .str.title()
            )

        logger.info("Text Standardization Completed.")
        
        # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------

    def feature_engineering(self):

        logger.info("=" * 70)
        logger.info("Feature Engineering")
        logger.info("=" * 70)

        current_year = pd.Timestamp.now().year

        self.df["Movie_Age"] = current_year - self.df["Year"]

        self.df["Genre_Count"] = (
            self.df["Genre"]
            .astype(str)
            .apply(lambda x: len(x.split(",")))
        )

        self.df["Is_MultiGenre"] = (
            self.df["Genre_Count"] > 1
        ).astype(int)

        logger.info("Feature Engineering Completed.")

    # -------------------------------------------------------------------------
    # Dataset Summary
    # -------------------------------------------------------------------------

    def dataset_summary(self):

        logger.info("=" * 70)
        logger.info("Dataset Summary")
        logger.info("=" * 70)

        logger.info(f"Rows               : {len(self.df)}")
        logger.info(f"Columns            : {len(self.df.columns)}")
        logger.info(f"Average Rating     : {self.df['Rating'].mean():.2f}")
        logger.info(f"Average Duration   : {self.df['Duration'].mean():.2f}")
        logger.info(f"Average Votes      : {self.df['Votes'].mean():.2f}")

    # -------------------------------------------------------------------------
    # Save Dataset
    # -------------------------------------------------------------------------

    def save_dataset(self):

        logger.info("=" * 70)
        logger.info("Saving Processed Dataset")
        logger.info("=" * 70)

        PROCESSED_DATA.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.df.to_csv(
            PROCESSED_DATA,
            index=False
        )

        logger.info(f"Saved Successfully")
        logger.info(PROCESSED_DATA)

    # -------------------------------------------------------------------------
    # Complete Pipeline
    # -------------------------------------------------------------------------

    def run(self):

        try:

            self.load_dataset()

            self.validate_dataset()

            self.dataset_information()

            self.remove_duplicates()

            self.handle_missing_values()

            self.clean_year()

            self.clean_duration()

            self.clean_votes()

            self.clean_rating()

            self.standardize_text_columns()

            self.feature_engineering()

            self.dataset_summary()

            self.save_dataset()

            logger.info("=" * 70)
            logger.info("Preprocessing Completed Successfully")
            logger.info("=" * 70)

            return self.df

        except Exception as error:

            logger.exception(
                f"Preprocessing Failed : {error}"
            )

            raise


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    processor = MoviePreprocessor()

    processor.run()