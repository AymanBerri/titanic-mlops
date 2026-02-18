import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="ML model for predicting Titanic passenger survival",
    version="1.0.0",
)


# Define request/response schemas
class PassengerFeatures(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str

    class Config:
        schema_extra = {
            "example": {
                "pclass": 3,
                "sex": "male",
                "age": 22.0,
                "sibsp": 1,
                "parch": 0,
                "fare": 7.25,
                "embarked": "S",
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    survival_status: str

    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.65,
                "survival_status": "Did not survive",
            }
        }


# Load model at startup
model = None
model_path = None


@app.on_event("startup")
async def load_model():
    """Load the latest trained model from MLflow"""
    global model

    try:
        # Find the latest run in MLflow
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name("titanic_baseline")

        if experiment is None:
            raise Exception("No experiment found. Train the model first.")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            raise Exception("No runs found. Train the model first.")

        # Load model from the latest run
        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        print(f"✅ Model loaded from run: {run_id}")

    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        print("Make sure to train the model first with: python src/train.py")
        model = None


@app.get("/")
async def root():
    return {
        "message": "Titanic Survival Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "unhealthy", "model_loaded": False}
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PassengerFeatures):
    """Predict survival for a single passenger"""
    global model

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first with: python src/train.py",
        )

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])

        # Apply same preprocessing as training
        # Encode categorical variables
        input_data["sex_male"] = (input_data["sex"] == "male").astype(int)

        # One-hot encode embarked
        embarked_dummies = pd.get_dummies(input_data["embarked"], prefix="embarked")
        for col in ["embarked_Q", "embarked_S"]:
            if col not in embarked_dummies.columns:
                embarked_dummies[col] = 0

        # Combine features
        features_df = pd.concat(
            [
                input_data[["pclass", "age", "sibsp", "parch", "fare"]],
                input_data[["sex_male"]],
                embarked_dummies,
            ],
            axis=1,
        )

        # Ensure columns are in the right order
        expected_columns = [
            "pclass",
            "age",
            "sibsp",
            "parch",
            "fare",
            "sex_male",
            "embarked_Q",
            "embarked_S",
        ]

        # Add any missing columns with 0
        for col in expected_columns:
            if col not in features_df.columns:
                features_df[col] = 0

        features_df = features_df[expected_columns]

        # Make prediction
        prediction = model.predict(features_df)[0]

        # Get probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_df)[0]
            probability = float(proba[1])  # Probability of survival
        else:
            probability = float(prediction)

        # Create response
        response = PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            survival_status="Survived" if prediction == 1 else "Did not survive",
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
