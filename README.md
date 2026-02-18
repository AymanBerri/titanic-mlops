# Titanic Survival Prediction - MLOps Project

## Project Description
This project predicts the survival of Titanic passengers using machine learning. The focus is not on model complexity but on applying MLOps best practices, including reproducible experiments, model tracking, containerization, and API-based deployment. Built as part of the MLOps course final project.

## Task
Binary classification: predict whether a passenger survived (1) or did not survive (0) based on passenger features like class, sex, age, and embarkation port.

## Dataset
Titanic dataset from Kaggle: https://www.kaggle.com/c/titanic/data

The dataset is stored in `data/raw/train.csv` and processed into `data/processed/train_processed.csv`.

## Project Structure
- `data/raw` : original CSV files  
- `data/processed` : cleaned and preprocessed data  
- `src/` : source code including preprocessing, training, prediction, and utility scripts  
- `tests/` : unit tests for preprocessing, training, and API  
- `mlruns/` : MLflow experiment tracking with model artifacts  
- `Dockerfile.train` : container for training  
- `Dockerfile.inference` : container for inference API  
- `pyproject.toml` and `uv.lock` : environment and dependency management  

## Team Members
- Coming Soon :/

---

## Checkpoint 1: Project Setup & Foundations 

- GitHub repository created with team write access
- Python environment managed with UV (`pyproject.toml`, `uv.lock`)
- Modular project structure with `src/`, `tests/`, `data/`
- Data loading and preprocessing logic implemented
- Baseline training script with Logistic Regression
- Clear README with project description and team roles

---

## Checkpoint 2: Code Quality & Experiment Tracking ✓

- **Pre-commit hooks**: black, isort, flake8 configured for code consistency
- **Unit tests**: 3 passing tests covering data preprocessing and training pipeline (>60% coverage)
- **MLflow integration**: Parameters, metrics (accuracy, F1), and model artifacts logged
- **Experiment tracking**: Clear experiment naming (`titanic_baseline`) with run comparison
- **Git history**: Meaningful, incremental commits showing project evolution

---

## Checkpoint 3: Model Serving & Containerization ✓

### FastAPI Inference Service
A production-ready API service built with FastAPI that serves the trained model:

- **POST /predict** - Accepts passenger features and returns survival prediction
- **GET /health** - Health check endpoint to verify model loading
- **GET /** - API information with documentation link

### API Schema

**Request Example:**

{
  "pclass": 3,
  "sex": "male",
  "age": 22.0,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S"
}

**Response Example:**

{
  "prediction": 0,
  "probability": 0.105,
  "survival_status": "Did not survive"
}

### Docker Containerization
- **Dockerfile.train**: Container for reproducible model training
- **Dockerfile.inference**: Container for the FastAPI inference service
- Volume mounting for MLflow artifacts to access trained models

### Running with Docker

**Build the inference image:**

docker build -f Dockerfile.inference -t titanic-inference:latest .

**Run the container:**

docker run -p 8000:8000 -v ${PWD}/mlruns:/app/mlruns titanic-inference:latest

**Test the API:**

python tests/test_api.py

### API Testing
A dedicated test script validates both health check and prediction endpoints:

python tests/test_api_simple.py

**Expected output:**

Status Code: 200
Response JSON: {'prediction': 0, 'probability': 0.105, 'survival_status': 'Did not survive'}

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| MLflow experiment corruption (missing meta.yaml) | Cleaned corrupted experiments and retrained with proper logging |
| Feature mismatch between training and prediction | Aligned preprocessing to use consistent 8 features across both pipelines |
| Docker volume mounting issues on Windows | Used absolute paths with proper Windows syntax for volume mounts |
| Model not found in expected path | Implemented dynamic path finding that scans artifacts directory |
| Flake8 linting errors on long lines | Refactored code to comply with 88-character line limit |

### Key Learnings
- MLflow tracking requires consistent artifact paths
- Feature engineering must be identical in training and serving
- Docker on Windows requires careful volume mount syntax
- Pre-commit hooks enforce code quality but require iterative fixes
- Team collaboration via GitHub with co-authored commits ensures proper credit

---

## Checkpoint 4: Monitoring & Final Report (Upcoming)

- Implement basic monitoring strategy
- Add request logging and metrics
- Prepare final project report
- Create demo video (≤10 minutes)
- CI/CD pipeline integration

---

## How to Run (Complete Pipeline)

### 1. Environment Setup
uv sync

### 2. Data Preprocessing
python src/data_preprocessing.py

### 3. Train Model
python src/train.py

### 4. View MLflow Experiments
mlflow ui
Open http://127.0.0.1:5000

### 5. Run Unit Tests
pytest --maxfail=1 --disable-warnings -q

### 6. Check Test Coverage
pytest --cov=src --cov-report=term-missing

### 7. Run API Locally (without Docker)
uvicorn src.predict:app --reload

### 8. Run API with Docker
docker build -f Dockerfile.inference -t titanic-inference:latest .
docker run -p 8000:8000 -v ${PWD}/mlruns:/app/mlruns titanic-inference:latest

### 9. Test API
python tests/test_api_simple.py

---

## Pre-commit Hooks
Pre-commit hooks ensure code quality on every commit:
- **black**: Automatic code formatting
- **isort**: Import sorting
- **flake8**: Linting with line length limit (88 characters)

---

## MLflow Tracking
MLflow tracks all experiments with:
- **Parameters**: Model type, hyperparameters
- **Metrics**: Accuracy, F1 score
- **Artifacts**: Trained model files
- **Experiment name**: `titanic_baseline`

---

## Git Workflow
- Feature branches with pull requests
- Meaningful commit messages
- Co-authored commits to credit team members
- Pre-commit hooks run automatically before commits

---

## Acknowledgments
This project was developed as part of the MLOps course (Jan 5, 2026 - Mar 15, 2026). Special thanks to the course instructors for guidance on MLOps best practices.