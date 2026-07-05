"""
===============================================================================
                            CineRate AI
                        UI Components Module
===============================================================================
"""

import streamlit as st


class UIComponents:

    @staticmethod
    def hero():

        st.markdown(
            """
<div class="hero">

<h1>🎬 CineRate AI</h1>

<h3>Professional Movie Rating Prediction Platform</h3>

<p>
Predict IMDb ratings using Machine Learning,
explore movie analytics,
and discover industry insights.
</p>

</div>
""",
            unsafe_allow_html=True,
        )

    @staticmethod
    def section(title, icon="📊"):

        st.markdown(
            f"""
<div class="section-title">
<h2>{icon} {title}</h2>
</div>
""",
            unsafe_allow_html=True,
        )

    @staticmethod
    def footer():

        st.markdown(
            """
<div class="footer">

<h2>🎬 CineRate AI</h2>

<p>Built using Python, Streamlit, Plotly & XGBoost</p>

<p style="opacity:.7;">
Developed for Data Science Portfolio
</p>

</div>
""",
            unsafe_allow_html=True,
        )