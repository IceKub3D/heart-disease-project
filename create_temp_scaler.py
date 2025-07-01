import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('data/heart_disease.csv')
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Create and fit scaler
scaler = StandardScaler()
scaler.fit(df[numerical_cols])

# Save scaler
joblib.dump(scaler, 'models/scaler.joblib')
print("Temporary scaler sav ed to models/scaler.joblib")