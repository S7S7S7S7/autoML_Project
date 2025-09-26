import streamlit as st
import numpy as np
import pickle
import os

def load_model_and_predict(model_name):
    model_filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
    if os.path.exists(model_filename):
        with open(model_filename, "rb") as f:
            model_data = pickle.load(f)
        
        model, scaler, feature_columns, data, mapping_dict = (
            model_data["model"], model_data["scaler"], model_data["features"],
            model_data["data"], model_data["mapping"]
        )

        input_data = []
        for feature in feature_columns:
            if feature in mapping_dict:
                options = list(mapping_dict[feature].keys())
                value = st.selectbox(f"Select {feature}", options)
                input_data.append(mapping_dict[feature][value])
            else:
                default_val = float(data[feature].mean())
                value = st.number_input(f"Enter {feature}", value=default_val)
                input_data.append(value)

        if st.button("Predict Value 🔮"):
            input_array = np.array(input_data).reshape(1, -1)
            if scaler is not None and hasattr(scaler, "transform"):
                input_array = scaler.transform(input_array)

            prediction = model.predict(input_array)
            if model_name == "K-means":
                st.success(f"Predicted Cluster: {prediction[0]}")
            else:
                st.success(f"Predicted Value: {prediction[0]:.2f}")
