import numpy as np
import pandas as pd

__all__ = ['create_heart_disease_features']

def create_heart_disease_features(df):
    """
    Generates engineered features to improve model performance for heart disease prediction.
    """
    df = df.copy()

    numeric_cols = ['age', 'thalach', 'trestbps', 'chol', 'cp', 'sex', 'exang', 'oldpeak']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ratio of maximum heart rate achieved vs. theoretical maximum
    df['hr_achievement'] = df['thalach'] / (220 - df['age'])
    # Absolute deviation from a healthy baseline (120 mmHg)
    df['bp_risk'] = (df['trestbps'] - 120).abs()
    # Binary indicator for high cholesterol (Threshold > 200)
    df['chol_risk'] = np.where(df['chol'] > 200, 1, 0)
    # Explores how cholesterol risk scales with age
    df['age_chol_interaction'] = df['age'] * df['chol'] / 1000
    # Inverts 'cp' (where 4 is often asymptomatic/less severe) 
    # to create an increasing scale of severity
    df['chest_pain_severity'] = 4 - df['cp']
    # A simplified count of cardiovascular risk factors 
    # based on standard medical thresholds
    df['combined_risk'] = (
        (df['age'] > 55).astype(int) + 
        (df['sex'] == 1).astype(int) +
        (df['chol'] > 240).astype(int) +
        (df['trestbps'] > 140).astype(int) +
        df['exang']
    )
    # Weights the ST depression (oldpeak) by the patient's age
    df['oldpeak_age_adjusted'] = df['oldpeak'] * (df['age'] / 50)
    
    return df