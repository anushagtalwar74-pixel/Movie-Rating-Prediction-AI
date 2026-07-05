"""
===========================================================
CineRate AI - Configuration File
Author : Anusha G Talawar
Project : Movie Rating Prediction using Machine Learning
===========================================================
"""

from pathlib import Path

# ===============================
# Project Directories
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# ===============================
# Dataset Paths
# ===============================

RAW_DATA = DATA_DIR / "IMDb Movies India.csv"
PROCESSED_DATA = DATA_DIR / "processed_movies.csv"

# ===============================
# Model Paths
# ===============================

MODEL_FILE = MODEL_DIR / "movie_model.pkl"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.pkl"

# ===============================
# Project Information
# ===============================

APP_NAME = "🎬 CineRate AI"

APP_TITLE = "AI Movie Rating Prediction Platform"

AUTHOR = "Anusha Talawar"

VERSION = "1.0"

# ===============================
# Theme Colors
# ===============================

PRIMARY = "#FF4B4B"

SECONDARY = "#6C63FF"

SUCCESS = "#00C853"

WARNING = "#FF9800"

BACKGROUND = "#0E1117"

CARD = "#1E1E2F"

TEXT = "#FFFFFF"

# ===============================
# Machine Learning
# ===============================

TARGET_COLUMN = "Rating"

RANDOM_STATE = 42

TEST_SIZE = 0.20

# ===============================
# Dashboard
# ===============================

TOP_DIRECTORS = 10

TOP_ACTORS = 10

TOP_MOVIES = 10

# ===============================
# Prediction
# ===============================

DEFAULT_DURATION = 120

DEFAULT_YEAR = 2020

DEFAULT_VOTES = 1000
