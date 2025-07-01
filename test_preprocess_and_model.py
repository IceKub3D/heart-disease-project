import pandas as pd
import joblib
import pytest
import numpy as np
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

    # Expected columns after preprocessing
    expected_columns = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex', 'fbs', 'exang', 'ca',
        'cp_1.0', 'cp_2.0', 'cp_3.0', 'cp_4.0',
        'restecg_0.0', 'restecg_0.4', 'restecg_0.6', 'restecg_1.0', 'restecg_2.0',
        'slope_1.0', 'slope_1.2', 'slope_1.4', 'slope_1.6', 'slope_1.8', 'slope_2.0',
        'slope_2.2', 'slope_2.4', 'slope_2.6', 'slope_3.0',
        'thal_3.0', 'thal_3.8', 'thal_4.2', 'thal_4.4', 'thal_4.6', 'thal_5.0',
        'thal_5.2', 'thal_5.4', 'thal_5.8', 'thal_6.0', 'thal_6.2', 'thal_6.4',
        'thal_6.6', 'thal_6.8', 'thal_7.0',
        'source_cleveland', 'source_hungarian', 'source_switzerland', 'source_va_long_beach',
        'thalach_exang'
    ]

    # Test: Check if all expected columns are present
    assert all(col in processed_data.columns for col in expected_columns), f"Missing columns: {[col for col in expected_columns if col not in processed_data.columns]}"

    # Test: Check if numerical columns are scaled
    assert processed_data['age'].iloc[0] != 63, f"Age should be scaled, got {processed_data['age'].iloc[0]}"
    assert processed_data['trestbps'].iloc[0] != 145, f"Trestbps should be scaled, got {processed_data['trestbps'].iloc[0]}"
    assert not np.isclose(processed_data['age'].iloc[0], 63, rtol=1e-5), f"Age should be significantly different, got {processed_data['age'].iloc[0]}"

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
    processed_data = processed_data.drop(columns=['target'])

    # Ensure model features are present
    model_features = getattr(model, 'feature_names', [
        'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
        'cp_2.0', 'cp_3.0', 'cp_4.0',
        'restecg_0.4', 'restecg_0.6', 'restecg_1.0', 'restecg_2.0',
        'slope_1.2', 'slope_1.4', 'slope_1.6', 'slope_1.8', 'slope_2.0', 'slope_2.2',
        'slope_2.4', 'slope_2.6', 'slope_3.0',
        'thal_3.8', 'thal_4.2', 'thal_4.4', 'thal_4.6', 'thal_5.0', 'thal_5.2',
        'thal_5.4', 'thal_5.8', 'thal_6.0', 'thal_6.2', 'thal_6.4', 'thal_6.6',
        'thal_6.8', 'thal_7.0',
        'source_hungarian', 'source_switzerland', 'source_va_long_beach',
        'thalach_exang'
    ])
    for col in model_features:
        if col not in processed_data.columns:
            processed_data[col] = 0.0

    # Ensure columns match model features exactly
    processed_data = processed_data[model_features]

    # Predict
    prediction = model.predict_proba(processed_data)[:, 1]  # Probability of positive class

    # Test: Check if prediction is a valid probability
    assert 0 <= prediction[0] <= 1, f"Prediction {prediction[0]} is not a valid probability"