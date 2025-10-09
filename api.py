# api.py
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, create_model
import os
from pathlib import Path
import glob

# --- 1. Definitive Model Loading (Bypassing the DB) ---
# This method finds the model file directly on the container's filesystem.
# It is robust and independent of the host machine's paths.
print("Searching for the latest MLflow run to load the model...")

# The Dockerfile copies everything to /app, so our mlruns folder is at /app/mlruns
# For local testing, it's in the current directory.
mlruns_path = Path("./mlruns")

# Find the latest experiment ID (highest number)
experiment_ids = [d.name for d in mlruns_path.iterdir() if d.is_dir() and d.name.isdigit()]
latest_experiment_id = sorted(experiment_ids, key=int, reverse=True)[0]

# Find the latest run ID inside that experiment
latest_run_path = max(
    (mlruns_path / latest_experiment_id).glob('*'),
    key=os.path.getmtime
)
run_id = latest_run_path.name

# Construct the final path to the model artifact
model_path = latest_run_path / "artifacts" / "review-score-model"
print(f"Found latest model in path: {model_path}")

# Load the model directly from this path
model = mlflow.pyfunc.load_model(str(model_path))
print("Model loaded successfully.")


# --- 2. Dynamically Create the Input Schema ---
input_schema = model.metadata.get_input_schema()
feature_names = [col.name for col in input_schema.inputs]
type_map = {'int64': int, 'float64': float, 'bool': bool, 'object': str}
feature_types = [type_map.get(str(col.type.to_pandas()), str) for col in input_schema.inputs]

pydantic_fields = {name: (dtype, ...) for name, dtype in zip(feature_names, feature_types)}
CustomerFeatures = create_model('CustomerFeatures', **pydantic_fields)


# --- 3. Create the FastAPI app ---
app = FastAPI(title="Customer Review Score API", version="1.0")

@app.post("/predict", tags=["Prediction"])
def predict(features: CustomerFeatures):
    """
    Receives order features and returns a predicted review score (1-5).
    """
    input_df = pd.DataFrame([features.dict()], columns=feature_names)
    prediction = model.predict(input_df)[0] + 1
    return {"predicted_review_score": int(prediction)}

@app.get("/", tags=["Health Check"])
def read_root():
    """Root endpoint for health check."""
    return {"message": "API is running. Go to /docs for interactive documentation."}