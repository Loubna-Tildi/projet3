import streamlit as st


def styles():
    """Applique un style CSS personnalisé à l'application Streamlit."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
        }
        h1 {
            color: #16223B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
