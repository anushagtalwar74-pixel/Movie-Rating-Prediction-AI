"""
Dashboard Module
"""

import streamlit as st


class Dashboard:

    @staticmethod
    def metrics(stats):

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "🎥 Movies",
                stats["Total Movies"]
            )

        with c2:
            st.metric(
                "⭐ Avg Rating",
                stats["Average Rating"]
            )

        with c3:
            st.metric(
                "🎬 Directors",
                stats["Total Directors"]
            )

        with c4:
            st.metric(
                "👥 Actors",
                stats["Total Actors"]
            )