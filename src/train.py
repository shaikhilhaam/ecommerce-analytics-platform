# src/train.py
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb
import os
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import yaml

def main():
    """
    Main function to train the review score prediction model with a definitive,
    robust MLflow configuration that manually saves and logs the model artifact.
    """
    
    # --- MLflow Setup: Rely on the default 'mlruns' directory ---
    experiment_name = "Review Score Prediction"
    mlflow.set_experiment(experiment_name=experiment_name)

    # --- Data Loading and Preparation ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_path = PROJECT_ROOT / "data" / "processed" / "propensity_dataset_final.csv"
    df = pd.read_csv(data_path)
    
    df.dropna(subset=['review_score'], inplace=True)
    df.drop(columns=['is_repeat'], inplace=True)
    df['review_score'] = df['review_score'] - 1

    X = df.drop(columns=['customer_unique_id', 'review_score'])
    y = df['review_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Model Training ---
    with mlflow.start_run(run_name="XGBoost_Multiclass_Classifier_Final") as run:
        print("--- Starting Final Model Training ---")
        
        params = {
            'objective': 'multi:softmax', 'num_class': 5, 'n_estimators': 300,
            'max_depth': 7, 'learning_rate': 0.1, 'eval_metric': 'mlogloss'
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Final Model Metrics: Accuracy={accuracy:.3f}, Weighted-F1={f1_weighted:.3f}")
        
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1_weighted)
        
        # --- DEFINITIVE FIX: Manually save the model locally first ---
        print("Manually saving model to a temporary local directory...")
        temp_model_dir = "temp_model_dir"
        os.makedirs(temp_model_dir, exist_ok=True)
        model_path = os.path.join(temp_model_dir, "model.xgb")
        model.save_model(model_path) # Use the raw XGBoost save method

        # Create the MLmodel file manually
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlmodel_content = {
            "flavors": {
                "python_function": {
                    "loader_module": "mlflow.xgboost",
                    "python_version": "3.11", # Adjust if your version is different
                    "data": "model.xgb",
                    "env": {"conda": "conda.yaml"}
                },
                "xgboost": {
                    "xgb_version": xgb.__version__,
                    "data": "model.xgb",
                    "model_class": "xgboost.XGBClassifier"
                }
            },
            "signature": signature.to_dict()
        }
        with open(os.path.join(temp_model_dir, "MLmodel"), 'w') as f:
            yaml.dump(mlmodel_content, f)

        # Create a simple conda.yaml
        conda_env = {
            'dependencies': [
                f'python=3.11',
                f'pip',
                {
                    'pip': [
                        f'mlflow=={mlflow.__version__}',
                        f'xgboost=={xgb.__version__}',
                        f'scikit-learn',
                        f'cloudpickle'
                    ]
                }
            ]
        }
        with open(os.path.join(temp_model_dir, "conda.yaml"), 'w') as f:
            yaml.dump(conda_env, f)

        # --- Use the reliable log_artifacts to upload the entire directory ---
        print("Logging the entire model directory as an artifact...")
        mlflow.log_artifacts(temp_model_dir, artifact_path="review-score-model")
        
        # Clean up the temporary directory
        shutil.rmtree(temp_model_dir)
        print("Final model logged successfully as a visible run artifact.")

        # (Feature importance plot logging remains the same)
        print("Generating and logging feature importance plot...")
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(model, ax=ax)
        feature_importance_path = "feature_importance.png"
        fig.savefig(feature_importance_path)
        plt.close(fig)
        mlflow.log_artifact(feature_importance_path, "plots")
        os.remove(feature_importance_path)
        print("Feature importance plot logged successfully.")

if __name__ == "__main__":
    main()