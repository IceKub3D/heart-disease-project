import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)


def preprocess_data(df):
    try:
        scaler = joblib.load('models/scaler.joblib')
        logging.info(f"Scaler loaded: {scaler}")
    except Exception as e:
        logging.error(f"Failed to load scaler: {e}")
        raise

    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    retain_cols = ['sex', 'fbs', 'exang', 'ca']

    # Ensure numerical columns are float
    for col in numerical_cols:
        df[col] = df[col].astype(float)

    # Scale numerical features
    try:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        logging.info(f"Scaled numerical columns: {numerical_cols}")
    except Exception as e:
        logging.error(f"Scaling failed: {e}")
        raise

    # Ensure categorical columns are float
    for col in categorical_cols:
        df[col] = df[col].astype(float)

    # Define all possible categories
    categories = {
        'cp': [1.0, 2.0, 3.0, 4.0],
        'restecg': [0.0, 0.4, 0.6, 1.0, 2.0],
        'slope': [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0],
        'thal': [3.0, 3.8, 4.2, 4.4, 4.6, 5.0, 5.2, 5.4, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0],
        'source': ['cleveland', 'hungarian', 'switzerland', 'va_long_beach']
    }

    # One-hot encode with all categories
    for col in categorical_cols + ['source']:
        # Ensure only valid categories are used
        df[col] = df[col].apply(lambda x: x if x in categories[col] else categories[col][0])
        df = pd.get_dummies(df, columns=[col], drop_first=False, dtype=float, prefix=col)
        # Add missing columns
        for cat in categories[col]:
            col_name = f"{col}_{float(cat)}" if col != 'source' else f"{col}_{cat}"
            if col_name not in df.columns:
                df[col_name] = 0.0

    # Drop unexpected columns
    valid_columns = numerical_cols + retain_cols + [f"{col}_{float(cat)}" for col in categorical_cols for cat in
                                                    categories[col]] + [f"source_{cat}" for cat in
                                                                        categories['source']] + ['thalach_exang']
    df = df[[col for col in df.columns if col in valid_columns or col == 'target']]

    # Add interaction term
    df['thalach_exang'] = df['thalach'] * df['exang']

    return df