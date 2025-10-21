import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from CSV."""
    return pd.read_csv(filepath)


def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Split dataset into train and test sets."""
    return train_test_split(df, test_size=test_size, random_state=random_state)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess dataset."""
    # Remove rows with NaN in age, gender, ethnicity
    df = df.dropna(subset=["age", "gender", "ethnicity"])

    # Fill height/weight NaNs with mean
    for col in ["height", "weight"]:
        df[col] = df[col].fillna(df[col].mean())

    # One-hot encoding for ethnicity
    df = pd.get_dummies(df, columns=["ethnicity"], prefix="ethnicity", drop_first=True)

    # Binary variable for gender
    df["gender_binary"] = df["gender"].apply(lambda x: 1 if str(x).lower().startswith("m") else 0)

    return df
