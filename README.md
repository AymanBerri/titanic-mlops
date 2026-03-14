# Titanic Survival Prediction - MLOps Project

![CI Pipeline](https://github.com/AymanBerri/titanic-mlops/actions/workflows/ci.yml/badge.svg)

## Project Description
This project predicts the survival of Titanic passengers using machine learning. The focus is not on model complexity but on applying MLOps best practices, including reproducible experiments, model tracking, containerization, and API-based deployment. Built as part of the MLOps course final project.

## Task
Binary classification: predict whether a passenger survived (1) or did not survive (0) based on passenger features like class, sex, age, and embarkation port.

## Dataset
Titanic dataset from Kaggle: https://www.kaggle.com/c/titanic/data

The dataset is stored in `data/raw/train.csv` and processed into `data/processed/train_processed.csv`.

## Project Structure
```
├── .github/workflows/      # CI/CD pipeline with GitHub Actions
├── data/                   # Raw and processed data
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── predict.py          # FastAPI application
│   ├── monitoring.py       # Monitoring system
│   └── utils.py
├── tests/                  # Unit and API tests
│   ├── test_data.py
│   ├── test_train.py
│   └── test_api.py
├── mlruns/                  # MLflow experiment tracking
├── Dockerfile.train         # Container for training
├── Dockerfile.inference     # Container for inference API
├── pyproject.toml           # Dependencies
└── uv.lock                  # Locked dependencies
```

## Team Members
- **Ayman Berri**: Project setup, environment management, preprocessing, training pipeline, MLflow integration, code reviews
- **Husnain Ali**: FastAPI implementation, model loading logic, API testing, Docker configuration
- **Muhammad Irfan**: Documentation, testing, final integration, README updates
- **Usama Lodhi**: Monitoring implementation, CI/CD pipeline, final report, drift detection, code reviews

---

## Checkpoint 1: Project Setup & Foundations ✓

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
- **GET /metrics** - Monitoring metrics endpoint
- **GET /drift-check** - Model drift detection endpoint

### API Schema

**Request Example:**
```
{
  "pclass": 3,
  "sex": "male",
  "age": 22.0,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S"
}
```
**Response Example:**
```
{
  "prediction": 0,
  "probability": 0.105,
  "survival_status": "Did not survive"
}
```
### Docker Containerization
- **Dockerfile.train**: Container for reproducible model training
- **Dockerfile.inference**: Container for the FastAPI inference service
- Volume mounting for MLflow artifacts to access trained models

### Running with Docker

**Build the inference image:**
```
docker build -f Dockerfile.inference -t titanic-inference:latest .
```

**Run the container:**
```
docker run -p 8000:8000 -v ${PWD}/mlruns:/app/mlruns titanic-inference:latest
```

**Test the API:**
```
python tests/test_api.py
```

### API Testing
A dedicated test script validates both health check and prediction endpoints:

python tests/test_api.py

**Expected output:**
```
Testing Titanic Survival Prediction API...
--------------------------------------------------
✓ Health check passed    
✓ Prediction test passed: {'prediction': 0, 'probability': 0.105, 'survival_status': 'Did not survive'}     
--------------------------------------------------
Tests passed: 2/2
✅ All tests passed!
```
---

## Checkpoint 4: Monitoring & CI/CD ✓

### Monitoring System
The API includes basic monitoring:
- Request count tracking
- Response time measurement
- Health check endpoint
- Model drift detection
- Logging to `monitoring.log`

Metrics are available at `/metrics` endpoint.

### CI/CD Pipeline with GitHub Actions
- Automated testing on every push to main
- Code style checks with black, flake8, isort
- Dependency installation with pip
- Test execution with pytest
- Status badge in README shows passing/failing

View the pipeline: `https://github.com/AymanBerri/titanic-mlops/actions`

---

## System Architecture

The system consists of:
- **Training Pipeline**: Preprocessing → Model Training → MLflow Tracking
- **Inference Service**: FastAPI → Model Loading → Prediction Endpoint
- **Monitoring**: Request logging, response time tracking, drift detection
- **Containerization**: Docker for both training and inference
- **Experiment Tracking**: MLflow for parameter/metric logging
- **CI/CD**: GitHub Actions for automated testing

## MLOps Practices Implemented

1. **Reproducibility**: UV for dependency management, Docker containers
2. **Version Control**: Git with pre-commit hooks
3. **Experiment Tracking**: MLflow for parameters, metrics, artifacts
4. **Testing**: Unit tests with pytest (60%+ coverage)
5. **Model Serving**: FastAPI with clear schema
6. **Containerization**: Docker for consistent deployment
7. **Monitoring**: Request logging, performance metrics, drift detection
8. **CI/CD**: GitHub Actions workflow for automated testing

---

## How to Run (Complete Pipeline)

### 1. Environment Setup
`uv sync`

### 2. Data Preprocessing
`python src/data_preprocessing.py`

### 3. Train Model
`python src/train.py`

### 4. View MLflow Experiments
`mlflow ui`     
Open http://127.0.0.1:5000

### 5. Run Unit Tests
`pytest --maxfail=1 --disable-warnings -q`

### 6. Check Test Coverage
`pytest --cov=src --cov-report=term-missing`

### 7. Run API Locally (without Docker)
`uvicorn src.predict:app --reload`

### 8. Run API with Docker
`docker build -f Dockerfile.inference -t titanic-inference:latest .`     
`docker run -p 8000:8000 -v ${PWD}/mlruns:/app/mlruns titanic-inference:latest`

### 9. Test API
`python tests/test_api.py`

### 10. Check Monitoring Logs
`cat monitoring.log`

---

## Pre-commit Hooks
Pre-commit hooks ensure code quality on every commit:
- **black**: Automatic code formatting
- **isort**: Import sorting
- **flake8**: Linting with line length limit (88 characters)

## MLflow Tracking
MLflow tracks all experiments with:
- **Parameters**: Model type, hyperparameters
- **Metrics**: Accuracy, F1 score
- **Artifacts**: Trained model files
- **Experiment name**: `titanic_baseline`

## Git Workflow
- Feature branches with pull requests
- Meaningful commit messages
- Co-authored commits to credit team members
- PR reviews required before merging
- Pre-commit hooks run automatically before commits

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| MLflow experiment corruption (missing meta.yaml) | Cleaned corrupted experiments and retrained with proper logging |
| Feature mismatch between training and prediction | Aligned preprocessing to use consistent 8 features across both pipelines |
| Docker volume mounting issues on Windows | Used absolute paths with proper Windows syntax for volume mounts |
| Model not found in expected path | Implemented dynamic path finding that scans artifacts directory |
| Flake8 linting errors on long lines | Refactored code to comply with 88-character line limit |
| CI/CD failing due to missing dependencies | Added explicit pip install for all required packages |
| API tests failing in CI | Used mocks to test without running live server |
| MLflow permission issues in CI | Configured temp directory for MLflow in CI environment |

## Key Learnings
- MLflow tracking requires consistent artifact paths
- Feature engineering must be identical in training and serving
- Docker on Windows requires careful volume mount syntax
- Pre-commit hooks enforce code quality but require iterative fixes
- Team collaboration via GitHub with PR reviews ensures proper credit
- CI/CD pipelines need explicit dependency management
- Mocking is essential for testing in CI environments

---

## Limitations & Future Work

### Current Limitations
- Simple logistic regression model (baseline only)
- Basic monitoring without alerting
- No automated retraining
- No A/B testing capability
- Local MLflow tracking only (no remote server)

### Future Improvements
- Add more sophisticated models (Random Forest, XGBoost)
- Implement real-time alerting (Slack/Email)
- Set up automated retraining pipeline
- Add A/B testing framework
- Deploy to cloud (AWS/GCP/Azure)
- Add model versioning and canary deployments
- Set up remote MLflow tracking server
- Add more comprehensive monitoring dashboards

---

## Demo Video
Watch the demo: [Link to unlisted YouTube video - coming soon]

---

## Acknowledgments
This project was developed as part of the MLOps course (Jan 5, 2026 - Mar 15, 2026). Special thanks to the course instructors for guidance on MLOps best practices.
