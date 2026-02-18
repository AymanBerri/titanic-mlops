import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path="data/raw/train.csv"):
    df = pd.read_csv(path)  # use pandas to load the data
    return df


def preprocess(df):
    df = df.copy()

    # Fill numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    if "alive" in df.columns:
        df = df.drop(columns=["alive"])

    return df


def split_data(df):
    X = df.drop("survived", axis=1)
    y = df["survived"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    df = load_data()
    df_processed = preprocess(df)
    df_processed.to_csv("data/processed/train_processed.csv", index=False)
    print("✅ Data preprocessing complete.")
