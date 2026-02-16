# Titanic Survival Prediction

## Project Description
This project predicts the survival of Titanic passengers using machine learning. The focus is not on model complexity but on applying MLOps best practices, including reproducible experiments, model tracking, containerization, and API-based deployment.

## Task
Binary classification: predict whether a passenger survived (1) or did not survive (0).

## Dataset
Titanic dataset from Kaggle: https://www.kaggle.com/c/titanic/data

The dataset is stored in `data/raw/train.csv` and processed into `data/processed/train_processed.csv`.

## Project Structure
- `data/raw` : original CSV files  
- `data/processed` : cleaned and preprocessed data  
- `src/` : source code including preprocessing, training, prediction, and utility scripts  
- `tests/` : unit tests for preprocessing and training  
- `mlruns/` : MLflow experiment tracking  
- `Dockerfile.train` : container for training  
- `Dockerfile.inference` : container for inference  
- `pyproject.toml` and `uv.lock` : environment and dependency management  

## How to Run

### Data Preprocessing
python src/data_preprocessing.py

### Train Model with MLflow Tracking
python src/train.py

### View MLflow Experiments
mlflow ui  
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to view experiments and compare runs.

### Run Unit Tests
pytest --maxfail=1 --disable-warnings -q

### Check Test Coverage
pytest --cov=src --cov-report=term-missing  
- Ensures at least 60% coverage of preprocessing and training pipeline.

## Pre-commit Hooks
Pre-commit hooks are configured to ensure clean code and consistent style:
- **black** for formatting  
- **isort** for import order  
- **flake8** for linting  

Hooks run automatically on each commit to maintain code quality.

## MLflow
MLflow is used to track parameters, metrics, and trained models. Experiments are stored in `mlruns/`, ensuring reproducibility and easy comparison between runs.  
Current experiment name: `titanic_baseline`.

## Git Workflow
- Meaningful commits documenting each step:
  - environment locking with `uv.lock`  
  - pre-commit hook configuration  
  - unit tests updates  
  - MLflow integration  
  - formatting fixes with black/isort  

- Collaboration via GitHub with branch management and pull requests.

## Next Steps (Checkpoint 3 & Beyond)
- Build FastAPI service for model inference  
- Containerize training and inference with Docker  
- Setup CI/CD for automated testing and deployment  
- Add monitoring and logging for inference service  
- Polish README and project report for final submission
