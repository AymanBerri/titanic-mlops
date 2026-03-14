import os
import tempfile

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data_preprocessing import load_data, preprocess, split_data


def main():
    # Set MLflow tracking to use a temporary directory in CI
    if os.getenv("CI"):  # Check if running in GitHub Actions
        temp_dir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file:{temp_dir}")
        print(f"CI environment detected, using temp dir: {temp_dir}")
    else:
        mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment("titanic_baseline")

    df = load_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(df)

    with mlflow.start_run(run_name="logreg_run"):
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Log parameters and metrics
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 200)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc}, F1: {f1}")
        print("Model and metrics logged to MLflow")


if __name__ == "__main__":
    main()
