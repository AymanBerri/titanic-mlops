import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path="data/raw/train.csv"):
    """Load and standardize column names to lowercase."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    return df


def preprocess(df):
    """Clean data and create consistent feature set."""
    # Fill missing values
    df["age"].fillna(df["age"].median(), inplace=True)
    df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

    # Define the exact feature columns we want
    feature_cols = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "sex_male",
        "embarked_q",
        "embarked_s",
    ]

    # Ensure all exist (add missing as zeros)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Include target if present
    if "survived" in df.columns:
        return df[["survived"] + feature_cols]
    else:
        return df[feature_cols]


def split_data(df):
    """Split into X and y."""
    X = df.drop("survived", axis=1)
    y = df["survived"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
