"""
===============================================================================
                        CineRate AI Analytics Dashboard
===============================================================================
"""

import streamlit as st
import pandas as pd

from src.charts import MovieCharts


class Analytics:

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe

        self.chart = MovieCharts(self.df)

    def render(self):

        st.markdown("## 📊 Movie Analytics Dashboard")

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Overview",
                "🎬 Movies",
                "👨‍💼 Directors",
                "⭐ Actors"
            ]
        )

        # ===========================================================
        # OVERVIEW
        # ===========================================================

        with tab1:

            c1, c2 = st.columns(2)

            with c1:

                st.plotly_chart(
                    self.chart.rating_distribution(),
                    use_container_width=True
                )

            with c2:

                st.plotly_chart(
                    self.chart.genre_distribution(),
                    use_container_width=True
                )

            st.plotly_chart(
                self.chart.release_trend(),
                use_container_width=True
            )

        # ===========================================================
        # MOVIES
        # ===========================================================

        with tab2:

            c1, c2 = st.columns(2)

            with c1:

                st.plotly_chart(
                    self.chart.votes_vs_rating(),
                    use_container_width=True
                )

            with c2:

                st.plotly_chart(
                    self.chart.duration_vs_rating(),
                    use_container_width=True
                )

            st.dataframe(

                self.df.sort_values(
                    "Rating",
                    ascending=False
                ).head(20),

                use_container_width=True,

                height=500

            )

        # ===========================================================
        # DIRECTORS
        # ===========================================================

        with tab3:

            st.plotly_chart(

                self.chart.top_directors(),

                use_container_width=True

            )

        # ===========================================================
        # ACTORS
        # ===========================================================

        with tab4:

            st.plotly_chart(

                self.chart.top_actors(),

                use_container_width=True

            )