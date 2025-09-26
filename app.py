import streamlit as st
import pandas as pd
import os

from src.preprocessing import fill_missing, remove_outliers, preprocess_data
from src.visualization import display_correlation_heatmap
from src.model_training import train_and_save_model
from src.prediction import load_model_and_predict

st.set_page_config(page_title="AutoML Pipeline", layout="wide")
st.title("📊 AutoML - CSV Upload & ML Pipeline")

# Sidebar file upload
st.sidebar.header("Upload Your CSV File 📂")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview 📑")
    st.dataframe(data.head())

    # --- Preprocessing ---
    st.write("### Data Preprocessing 🛠️")
    if st.checkbox("Show missing values"):
        st.write(data.isnull().sum())

    if st.checkbox("Fill missing values with mean/mode"):
        data = fill_missing(data)
        st.write("✅ Missing values filled.")

    if st.checkbox("Remove outliers (IQR method)"):
        data = remove_outliers(data)
        st.success("✅ Outliers removed!")

    data, cat_cols, mapping_dict = preprocess_data(data)
    if cat_cols:
        st.write(f"Encoded categorical columns: {cat_cols}")

    # --- Visualization ---
    if st.checkbox("Show Correlation Heatmap"):
        display_correlation_heatmap(data)

    # --- Feature Selection ---
    st.write("### Feature Selection 🎯")
    target_column = st.selectbox("Select Target Column", data.columns)
    feature_columns = st.multiselect("Select Features", [col for col in data.columns if col != target_column])
    model_name = st.selectbox(
        "Select Model",
        ["Linear Regression", "Logistic Regression", "KNN", "Decision Tree", "Random Forest", "K-means"]
    )

    # --- Model Training ---
    if st.button("Train Model 🚀"):
        if not feature_columns:
            st.error("Please select at least one feature column!")
        else:
            model_filename = train_and_save_model(data, feature_columns, target_column, model_name, mapping_dict)
            st.success(f"✅ {model_name} trained & saved as {model_filename}")

    # --- Prediction Section ---
    st.write("### Make Predictions 🔮")
    load_model_and_predict(model_name)
