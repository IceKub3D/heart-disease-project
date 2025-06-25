
import pandas as pd
import joblib

def preprocess_data(df):
    scaler = joblib.load('models/scaler.joblib')
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    # Scale numerical features
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Encode categorical features
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df
