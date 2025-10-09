# api.py
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, create_model
from pathlib import Path

# --- Load the model from the MLflow Model Registry ---
MODEL_NAME = "review-score-model" # The name we will give our new model in the UI
MODEL_STAGE = "Production"

# --- MLflow Setup ---
# Use the robust pathing from train.py to find the database
PROJECT_ROOT = Path(__file__).resolve().parent
db_path = PROJECT_ROOT / "mlflow_data" / "mlflow.db"
tracking_uri = f"sqlite:///{db_path}"
mlflow.set_tracking_uri(tracking_uri)

model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

print(f"Loading model '{MODEL_NAME}' stage '{MODEL_STAGE}' from the registry...")
model = mlflow.pyfunc.load_model(model_uri)
print("Model loaded successfully.")

# --- Dynamically create the Pydantic model from the MLflow signature ---
input_schema = model.metadata.get_input_schema()
feature_names = [col.name for col in input_schema.inputs]
type_map = {'int64': int, 'float64': float}
feature_types = [type_map.get(str(col.type.to_pandas()), float) for col in input_schema.inputs]
pydantic_fields = {name: (dtype, ...) for name, dtype in zip(feature_names, feature_types)}
OrderFeatures = create_model('OrderFeatures', **pydantic_fields)

# --- Create the FastAPI app ---
app = FastAPI(title="Review Score Prediction API", version="1.0")

@app.post("/predict_review", tags=["Prediction"])
def predict_review(features: OrderFeatures):
    """
    Receives order features and returns a predicted review score (1-5).
    """
    input_df = pd.DataFrame([features.dict()], columns=feature_names)
    # The model predicts 0-4, so we add 1 to return a 1-5 score
    prediction = model.predict(input_df)[0] + 1
    return {"predicted_review_score": int(prediction)}

@app.get("/", tags=["Health Check"])
def read_root():
    return {"message": "API is running. Go to /docs for interactive documentation."}

