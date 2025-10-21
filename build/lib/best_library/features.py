import pandas as pd

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables (gender and ethnicity)."""
    # New column: binary variable for gender
    df['gender_binary'] = df['gender'].apply(lambda x: 1 if x == 'M' else 0)
    
    # One-hot encoding for ethnicity
    df = pd.get_dummies(df, columns=['ethnicity'], drop_first=True)
    return df