import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
# This script will automatically use the 'mlruns' directory in the current folder.
experiment_name = "Review Score Prediction"
registered_model_name = "ReviewScorePredictionModel"

# --- Initialize MLflow Client ---
client = MlflowClient()

# --- Find the Latest Run in Our Experiment ---
print(f"Searching for the latest run in experiment '{experiment_name}'...")
experiment = client.get_experiment_by_name(experiment_name)
latest_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)[0]

run_id = latest_run.info.run_id
print(f"Found latest run with ID: {run_id}")

# --- Construct the URI to the Model Artifact ---
# This points directly to the model folder inside the specific run.
artifact_path = "review-score-model"
model_uri = f"runs:/{run_id}/{artifact_path}"
print(f"Model artifact URI: {model_uri}")

# --- Register the Model ---
print(f"Registering the model as '{registered_model_name}'...")
model_version = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)
print(f"Model successfully registered as '{registered_model_name}' version {model_version.version}.")

# --- (Optional) Transition the Model to "Staging" ---
print("Transitioning the new model version to the 'Staging' stage...")
client.transition_model_version_stage(
    name=registered_model_name,
    version=model_version.version,
    stage="Staging"
)
print("Model transitioned to 'Staging'.")
