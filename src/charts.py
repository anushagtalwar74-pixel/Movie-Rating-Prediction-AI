"""
===============================================================================
                            CineRate AI
                         Interactive Plotly Charts
===============================================================================
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class MovieCharts:

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

        self.template = "plotly_dark"

    # ==========================================================
    # Rating Distribution
    # ==========================================================

    def rating_distribution(self):

        fig = px.histogram(
            self.df,
            x="Rating",
            nbins=20,
            title="Rating Distribution",
            template=self.template,
            color_discrete_sequence=["#00E5FF"]
        )

        fig.update_layout(height=420)

        return fig

    # ==========================================================
    # Genre Distribution
    # ==========================================================

    def genre_distribution(self):

        genres = (
            self.df["Genre"]
            .str.split(",")
            .explode()
            .str.strip()
            .value_counts()
            .head(10)
            .reset_index()
        )

        genres.columns = ["Genre", "Count"]

        fig = px.pie(

            genres,

            names="Genre",

            values="Count",

            hole=.55,

            title="Top Genres",

            template=self.template

        )

        fig.update_layout(height=420)

        return fig

    # ==========================================================
    # Release Trend
    # ==========================================================

    def release_trend(self):

        yearly = (

            self.df.groupby("Year")

            .size()

            .reset_index(name="Movies")

        )

        fig = px.line(

            yearly,

            x="Year",

            y="Movies",

            markers=True,

            title="Movies Released per Year",

            template=self.template

        )

        fig.update_layout(height=450)

        return fig

    # ==========================================================
    # Votes vs Rating
    # ==========================================================

    def votes_vs_rating(self):

        fig = px.scatter(

            self.df,

            x="Votes",

            y="Rating",

            color="Rating",

            title="Votes vs Rating",

            template=self.template,

            hover_data=["Name"]

        )

        fig.update_layout(height=450)

        return fig

    # ==========================================================
    # Duration vs Rating
    # ==========================================================

    def duration_vs_rating(self):

        fig = px.scatter(

            self.df,

            x="Duration",

            y="Rating",

            color="Duration",

            title="Duration vs Rating",

            template=self.template,

            hover_data=["Name"]

        )

        fig.update_layout(height=450)

        return fig

    # ==========================================================
    # Top Directors
    # ==========================================================

    def top_directors(self):

        directors = (

            self.df["Director"]

            .value_counts()

            .head(10)

            .reset_index()

        )

        directors.columns = ["Director", "Movies"]

        fig = px.bar(

            directors,

            x="Movies",

            y="Director",

            orientation="h",

            title="Top Directors",

            template=self.template,

            color="Movies"

        )

        fig.update_layout(height=500)

        return fig

    # ==========================================================
    # Top Actors
    # ==========================================================

    def top_actors(self):

        actors = pd.concat([

            self.df["Actor 1"],

            self.df["Actor 2"],

            self.df["Actor 3"]

        ])

        actors = (

            actors

            .value_counts()

            .head(10)

            .reset_index()

        )

        actors.columns = ["Actor", "Movies"]

        fig = px.bar(

            actors,

            x="Movies",

            y="Actor",

            orientation="h",

            title="Top Actors",

            template=self.template,

            color="Movies"

        )

        fig.update_layout(height=500)

        return fig

    # ==========================================================
    # Correlation Heatmap
    # ==========================================================

    def correlation_heatmap(self):

        corr = self.df[

            [

                "Year",

                "Duration",

                "Votes",

                "Rating"

            ]

        ].corr()

        fig = go.Figure(

            data=go.Heatmap(

                z=corr.values,

                x=corr.columns,

                y=corr.columns,

                text=corr.round(2),

                texttemplate="%{text}",

                colorscale="Viridis"

            )

        )

        fig.update_layout(

            title="Correlation Heatmap",

            template=self.template,

            height=500

        )

        return fig