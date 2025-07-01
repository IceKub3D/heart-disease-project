import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# Define expected model features
model_features = [
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
]

# Create a dummy XGBClassifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(np.array([[0] * len(model_features)]), [0])  # Minimal fit to initialize
model.feature_names = model_features  # Set feature names

# Save the model
joblib.dump(model, 'models/temp_model.joblib')
print("Temporary model saved to models/temp_model.joblib")