"""
===============================================================================
                            CineRate AI
                    Professional Visualization Module
-------------------------------------------------------------------------------
Author      : Anusha G Talawar
Project     : Movie Rating Prediction using Machine Learning
Version     : 2.0 Professional
===============================================================================
"""

import pandas as pd
import plotly.express as px
from src.config import *

class MovieVisualizer:

    def __init__(self):

        self.df = pd.read_csv(PROCESSED_DATA)

    # ---------------------------------------------------------------------
    # Rating Distribution
    # ---------------------------------------------------------------------

    def rating_distribution(self):

        fig = px.histogram(

            self.df,

            x="Rating",

            nbins=20,

            title="⭐ Rating Distribution",

            template="plotly_dark"

        )

        fig.update_layout(height=450)

        return fig

    # ---------------------------------------------------------------------
    # Top Rated Movies
    # ---------------------------------------------------------------------

    def top_movies(self, n=10):

        top = (

            self.df

            .sort_values(

                "Rating",

                ascending=False

            )

            .head(n)

        )

        fig = px.bar(

            top,

            x="Rating",

            y="Name",

            orientation="h",

            color="Rating",

            template="plotly_dark",

            title="🏆 Top Rated Movies"

        )

        fig.update_layout(height=500)

        return fig

    # ---------------------------------------------------------------------
    # Genre Distribution
    # ---------------------------------------------------------------------

    def genre_distribution(self):

        genre = (

            self.df["Genre"]

            .str.split(",")

            .explode()

            .str.strip()

            .value_counts()

            .head(15)

        )

        fig = px.bar(

            x=genre.index,

            y=genre.values,

            labels={

                "x": "Genre",

                "y": "Movies"

            },

            title="🎭 Genre Distribution",

            template="plotly_dark"

        )

        fig.update_layout(height=500)

        return fig
    
        # ---------------------------------------------------------------------
    # Top Directors
    # ---------------------------------------------------------------------

    def top_directors(self, n=10):

        directors = (

            self.df["Director"]

            .value_counts()

            .head(n)

            .sort_values()

        )

        fig = px.bar(

            x=directors.values,

            y=directors.index,

            orientation="h",

            labels={

                "x": "Movies",

                "y": "Director"

            },

            title="🎬 Top Directors",

            template="plotly_dark",

            color=directors.values,

            color_continuous_scale="Blues"

        )

        fig.update_layout(height=500)

        return fig

    # ---------------------------------------------------------------------
    # Top Actors
    # ---------------------------------------------------------------------

    def top_actors(self, n=10):

        actors = pd.concat([

            self.df["Actor 1"],

            self.df["Actor 2"],

            self.df["Actor 3"]

        ])

        actors = (

            actors

            .value_counts()

            .head(n)

            .sort_values()

        )

        fig = px.bar(

            x=actors.values,

            y=actors.index,

            orientation="h",

            labels={

                "x": "Movies",

                "y": "Actor"

            },

            title="⭐ Most Frequent Actors",

            template="plotly_dark",

            color=actors.values,

            color_continuous_scale="Oranges"

        )

        fig.update_layout(height=500)

        return fig

    # ---------------------------------------------------------------------
    # Movies Released Per Year
    # ---------------------------------------------------------------------

    def yearly_releases(self):

        yearly = (

            self.df

            .groupby("Year")

            .size()

            .reset_index(name="Movies")

        )

        fig = px.line(

            yearly,

            x="Year",

            y="Movies",

            markers=True,

            title="📈 Movies Released Per Year",

            template="plotly_dark"

        )

        fig.update_layout(height=450)

        return fig

    # ---------------------------------------------------------------------
    # Votes vs Rating
    # ---------------------------------------------------------------------

    def votes_vs_rating(self):

        fig = px.scatter(

            self.df,

            x="Votes",

            y="Rating",

            hover_data=["Name"],

            color="Rating",

            title="👍 Votes vs Rating",

            template="plotly_dark"

        )

        fig.update_layout(height=500)

        return fig
    
        # ---------------------------------------------------------------------
    # Correlation Heatmap
    # ---------------------------------------------------------------------

    def correlation_heatmap(self):

        corr = self.df[
            [
                "Year",
                "Duration",
                "Votes",
                "Rating",
                "Movie_Age",
                "Genre_Count"
            ]
        ].corr(numeric_only=True)

        fig = px.imshow(

            corr,

            text_auto=".2f",

            color_continuous_scale="Viridis",

            title="📊 Feature Correlation",

            template="plotly_dark"

        )

        fig.update_layout(height=550)

        return fig

    # ---------------------------------------------------------------------
    # Dashboard Statistics
    # ---------------------------------------------------------------------

    def dashboard_stats(self):

        stats = {

            "Total Movies": int(len(self.df)),

            "Average Rating": round(self.df["Rating"].mean(), 2),

            "Highest Rating": round(self.df["Rating"].max(), 2),

            "Average Duration": round(self.df["Duration"].mean(), 2),

            "Average Votes": int(self.df["Votes"].mean()),

            "Total Directors": int(self.df["Director"].nunique()),

            "Total Actors": int(
                pd.concat([
                    self.df["Actor 1"],
                    self.df["Actor 2"],
                    self.df["Actor 3"]
                ]).nunique()
            )

        }

        return stats

    # ---------------------------------------------------------------------
    # Highest Rated Movies Table
    # ---------------------------------------------------------------------

    def top_movies_table(self, n=15):

        table = (

            self.df

            .sort_values(

                "Rating",

                ascending=False

            )

            [

                [

                    "Name",

                    "Year",

                    "Genre",

                    "Director",

                    "Rating",

                    "Votes"

                ]

            ]

            .head(n)

        )

        return table

        # ---------------------------------------------------------------------
    # Model Performance Table
    # ---------------------------------------------------------------------

    def model_performance(self):

        data = pd.DataFrame({
            "Model": [
                "XGBoost",
                "Random Forest",
                "Gradient Boosting",
                "Linear Regression"
            ],
            "R² Score": [
                0.3956,
                0.3784,
                0.3607,
                0.2216
            ],
            "MAE": [
                0.7977,
                0.8053,
                0.8320,
                0.9421
            ],
            "RMSE": [
                1.0600,
                1.0750,
                1.0902,
                1.2030
            ]
        })

        return data

    # ---------------------------------------------------------------------
    # Model Comparison Chart
    # ---------------------------------------------------------------------

    def model_comparison(self):

        df = self.model_performance()

        fig = px.bar(
            df,
            x="Model",
            y="R² Score",
            color="R² Score",
            text="R² Score",
            title="🏆 Machine Learning Model Comparison",
            template="plotly_dark"
        )

        fig.update_traces(texttemplate="%{text:.3f}")

        fig.update_layout(
            height=500,
            xaxis_title="Model",
            yaxis_title="R² Score"
        )

        return fig

# =============================================================================
# Test Module
# =============================================================================

if __name__ == "__main__":

    visualizer = MovieVisualizer()

    print("=" * 60)

    print("Dashboard Statistics")

    print("=" * 60)

    stats = visualizer.dashboard_stats()

    for key, value in stats.items():

        print(f"{key:20}: {value}")

    print("=" * 60)