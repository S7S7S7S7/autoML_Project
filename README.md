# AutoML Streamlit App 🚀

An end-to-end AutoML pipeline built with Streamlit.  
Features:
- Upload CSV file
- Data preprocessing (missing values, outliers, encoding)
- Correlation heatmap
- Train multiple ML models
- Save & download trained models
- Make predictions with saved models

## 📂 Project Structure
- `app.py`: Streamlit main app
- `src/`: Source code (preprocessing, visualization, training, prediction)
- `models/`: Trained models
- `data/`: Sample dataset

# AutoML Streamlit Application 🚀

**AutoML Streamlit App** is an end-to-end machine learning pipeline that allows users to upload datasets, preprocess data, train multiple machine learning models, and make predictions — all through an easy-to-use web interface. This project is designed to demonstrate real-world industry practices in building, deploying, and managing machine learning models.

---

## **Key Features**

1. **User-Friendly Data Upload**  
   - Upload CSV files directly through the web interface.  
   - Preview dataset to quickly understand the structure and contents.  

2. **Data Preprocessing**  
   - Automatically handles missing values.  
   - Removes outliers to improve model performance.  
   - Encodes categorical variables for machine learning models.  

3. **Data Visualization**  
   - Displays correlation heatmaps to show relationships between features.  

4. **Feature & Target Selection**  
   - Choose target column and input features for modeling.  

5. **Multiple ML Models**  
   - **Regression:** Linear Regression, KNN, Decision Tree, Random Forest  
   - **Classification:** Logistic Regression  
   - **Clustering:** K-Means  
   - Automatic scaling applied where necessary.  

6. **Model Evaluation**  
   - Regression metrics: MAE, MSE, R² Score  
   - Classification metrics: Accuracy  
   - Clustering results: Cluster Centers  

7. **Model Persistence & Deployment**  
   - Trained models are saved automatically using Pickle.  
   - Load saved models to make predictions on new data.  
   - Streamlit interface allows interactive prediction without coding.  

---