import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def display_correlation_heatmap(data):
    st.write("### Correlation Heatmap 📉")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
