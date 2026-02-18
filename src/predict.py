import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Titanic Survival Prediction API", version="1.0.0")


class PassengerFeatures(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str

    class Config:
        json_schema_extra = {
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
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.65,
                "survival_status": "Did not survive",
            }
        }


model = None

# The exact feature set used during training
EXPECTED_FEATURES = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "sex_male",
    "embarked_q",
    "embarked_s",
]


@app.on_event("startup")
async def load_model():
    global model
    try:
        # Path to the latest run (auto-detected as before)
        base = Path("/app/mlruns")
        experiments = [d for d in base.iterdir() if d.is_dir()]
        for exp in experiments:
            models_dir = exp / "models"
            if models_dir.exists():
                runs = [
                    d
                    for d in models_dir.iterdir()
                    if d.is_dir() and d.name.startswith("m-")
                ]
                if runs:
                    latest_run = max(runs, key=lambda p: p.stat().st_mtime)
                    artifacts = latest_run / "artifacts"
                    if artifacts.exists():
                        # Try loading from the artifacts folder
                        model = mlflow.sklearn.load_model(str(artifacts))
                        logger.info(f"✅ Model loaded from {artifacts}")
                        # Log expected features for debugging
                        if hasattr(model, "feature_names_in_"):
                            logger.info(f"Model expects features: \
                                {model.feature_names_in_.tolist()}")
                        return
        logger.error("No model could be loaded.")
    except Exception:
        logger.exception("Model loading failed")
        model = None


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PassengerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build a DataFrame with the exact expected features
        input_dict = features.dict()
        # Start with a dictionary of zeros for all expected features
        data = {col: 0 for col in EXPECTED_FEATURES}
        # Fill in the ones we have
        data["pclass"] = input_dict["pclass"]
        data["age"] = input_dict["age"]
        data["sibsp"] = input_dict["sibsp"]
        data["parch"] = input_dict["parch"]
        data["fare"] = input_dict["fare"]
        data["sex_male"] = 1 if input_dict["sex"].lower() == "male" else 0
        # Embarked: one-hot (embarked_q, embarked_s)
        if input_dict["embarked"].upper() == "Q":
            data["embarked_q"] = 1
        elif input_dict["embarked"].upper() == "S":
            data["embarked_s"] = 1
        # 'C' leaves both as zero

        input_df = pd.DataFrame([data])

        # Ensure column order matches model's expectation (if available)
        if hasattr(model, "feature_names_in_"):
            input_df = input_df[model.feature_names_in_]
        else:
            input_df = input_df[EXPECTED_FEATURES]

        logger.info(f"Input features: {input_df.to_dict()}")

        pred = model.predict(input_df)[0]
        proba = (
            model.predict_proba(input_df)[0][1]
            if hasattr(model, "predict_proba")
            else float(pred)
        )

        return PredictionResponse(
            prediction=int(pred),
            probability=float(proba),
            survival_status="Survived" if pred == 1 else "Did not survive",
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
