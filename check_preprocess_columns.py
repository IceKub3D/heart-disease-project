import pandas as pd
from preprocess import preprocess_data

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

# Print actual columns
print("Actual columns:", processed_data.columns.tolist())