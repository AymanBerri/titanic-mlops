# import joblib
import mlflow
import mlflow.sklearn

# import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .data_preprocessing import load_data, preprocess, split_data


def main():
    # MLflow experiment:
    # Clear experiment naming and comparison
    # (all runs go into one experiment)
    mlflow.set_experiment("titanic_baseline")

    df = load_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # MLflow integrated with log parameters, metrics, model artifacts
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log metrics & model
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc}")
        print("Model and metrics logged to MLflow")


if __name__ == "__main__":
    main()
