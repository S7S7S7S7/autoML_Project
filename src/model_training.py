import os
import pickle
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

def train_and_save_model(data, feature_columns, target_column, model_name, mapping_dict):
    X = data[feature_columns]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler, X_train_scaled, X_test_scaled = StandardScaler(), None, None
    if model_name != "Decision Tree" and model_name != "Random Forest":
        X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

    model = None
    if model_name == "Linear Regression":
        model = LinearRegression().fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}, MSE: {mean_squared_error(y_test, y_pred):.2f}, R²: {r2_score(y_test, y_pred):.2f}")

    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
        st.write(f"Accuracy: {accuracy_score(y_test, model.predict(X_test_scaled))*100:.2f}%")

    elif model_name == "KNN":
        model = KNeighborsRegressor().fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}, R²: {r2_score(y_test, y_pred):.2f}")

    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor().fit(X_train, y_train)
        st.write(f"MSE: {mean_squared_error(y_test, model.predict(X_test)):.2f}")

    elif model_name == "Random Forest":
        model = RandomForestRegressor().fit(X_train, y_train)
        st.write(f"MSE: {mean_squared_error(y_test, model.predict(X_test)):.2f}")

    elif model_name == "K-means":
        model = KMeans(n_clusters=3, random_state=42).fit(X_train)
        st.write("Cluster Centers:", model.cluster_centers_)

    os.makedirs("models", exist_ok=True)
    model_filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": feature_columns, "data": data, "mapping": mapping_dict}, f)
    
    return model_filename
