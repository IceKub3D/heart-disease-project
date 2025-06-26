import pandas as pd
import joblib
import pytest
from preprocess import preprocess_data


def test_preprocess_data():
    # Create a sample row based on heart_disease.csv columns
    sample_data = pd.DataFrame({
        'age': [63],
        'sex': [1],
        'cp': [3],
        'trestbps': [145],
        'chol': [233],
        'fbs': [1],
        'restecg': [0],
        'thalach': [150],
        'exang': [0],
        'oldpeak': [2.3],
        'slope': [0],
        'ca': [0],
        'thal': [1],
        'target': [1],
        'source': ['cleveland']
    })

    # Preprocess the sample data
    processed_data = preprocess_data(sample_data)

    # Expected columns after preprocessing (numerical + one-hot encoded categorical)
    expected_columns = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target', 'source',
        'sex_1', 'cp_1', 'cp_2', 'cp_3', 'fbs_1', 'restecg_1', 'restecg_2',
        'exang_1', 'slope_1', 'slope_2', 'ca_1', 'ca_2', 'ca_3', 'thal_1', 'thal_2', 'thal_3'
    ]

    # Test: Check if all expected columns are present
    assert all(col in processed_data.columns for col in expected_columns), "Missing expected columns"

    # Test: Check if numerical columns are scaled (not in original range)
    assert processed_data['age'].iloc[0] != 63, "Age should be scaled"
    assert processed_data['trestbps'].iloc[0] != 145, "Trestbps should be scaled"

    # Test: Check if categorical columns are one-hot encoded
    assert processed_data['sex_1'].iloc[0] == 1, "Sex encoding incorrect"


def test_model_inference():
    # Load model and scaler
    model = joblib.load('models/model.joblib')
    scaler = joblib.load('models/scaler.joblib')

    # Create a sample row
    sample_data = pd.DataFrame({
        'age': [63],
        'sex': [1],
        'cp': [3],
        'trestbps': [145],
        'chol': [233],
        'fbs': [1],
        'restecg': [0],
        'thalach': [150],
        'exang': [0],
        'oldpeak': [2.3],
        'slope': [0],
        'ca': [0],
        'thal': [1],
        'target': [1],
        'source': ['cleveland']
    })

    # Preprocess the sample data
    processed_data = preprocess_data(sample_data)

    # Drop non-feature columns
    processed_data = processed_data.drop(columns=['target', 'source'])

    # Predict
    prediction = model.predict_proba(processed_data)[:, 1]  # Probability of positive class

    # Test: Check if prediction is a valid probability
    assert 0 <= prediction[0] <= 1, f"Prediction {prediction[0]} is not a valid probability"
