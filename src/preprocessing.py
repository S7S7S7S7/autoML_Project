import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fill_missing(data):
    for col in data.columns:
        if data[col].dtype == "object":
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].mean(), inplace=True)
    return data

def remove_outliers(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

def preprocess_data(data):
    categorical_cols = [col for col in data.columns if data[col].dtype == "object"]
    encoder = LabelEncoder()
    mapping_dict = {}
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col].astype(str))
        mapping_dict[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    return data, categorical_cols, mapping_dict
