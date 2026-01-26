# Titanic Survival Prediction - MLOps Demo

## Project Description
This project predicts the survival of Titanic passengers using machine learning. The focus of this project is not on model complexity but on applying MLOps best practices, including reproducible experiments, model tracking, containerization, and API-based deployment.

## Task
Binary classification: predict whether a passenger survived (1) or did not survive (0).

## Dataset
Titanic dataset from Kaggle: https://www.kaggle.com/c/titanic/data

The dataset is stored in `data/raw/train.csv` and processed into `data/processed/train_processed.csv`.

## Project Structure
- `data/raw` : original CSV files  
- `data/processed` : cleaned and preprocessed data  
- `src/` : source code including preprocessing, training, prediction, and utility scripts  
- `tests/` : unit tests  
- `mlruns/` : MLflow experiment tracking  
- `Dockerfile.train` : container for training  
- `Dockerfile.inference` : container for inference  
- `pyproject.toml` and `uv.lock` : environment and dependency management  

## How to Run
To preprocess data:
python src/data_preprocessing.py

To train baseline model with MLflow tracking:
python src/train.py

To view experiments in MLflow UI:
mlflow ui

## MLflow
MLflow is used to track parameters, metrics, and trained models. All experiments are stored locally in `mlruns/`. This ensures reproducibility and easy comparison between runs.

## Next Steps
- Implement unit tests for preprocessing and training scripts  
- Configure pre-commit hooks for code quality  
- Build FastAPI service for model inference  
- Containerize training and inference using Docker  
- Setup CI/CD for automated testing and deployment  
- Add monitoring for inference service
