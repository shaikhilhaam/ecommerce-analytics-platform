# src/train.py
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import os

def train_model(X_train, X_test, y_train, y_test, params, run_name):
    """Trains a single XGBoost Classifier and logs it to MLflow under a specific run name."""
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Starting MLflow Run: {run.info.run_name}")

        # Use XGBClassifier for Yes/No (binary) prediction
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Run '{run_name}': Accuracy={accuracy:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

        # Log parameters and metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log the model with an input example
        input_example = X_train.head()
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost-propensity-model",
            input_example=input_example
        )

def main():
    """Main function to load data and run multiple training experiments."""
    mlflow.set_tracking_uri("sqlite:///mlflow_data/mlflow.db")
    mlflow.set_experiment("Repeat Purchase Propensity") # Renamed experiment for clarity

    # Load the new dataset for the classification task
    data_path = os.path.join('data', 'processed', 'propensity_dataset.csv')
    modeling_df = pd.read_csv(data_path)

    # Define features (X) and the new binary target (y)
    X = modeling_df.drop(columns=['customer_unique_id', 'is_repeat'])
    y = modeling_df['is_repeat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Experiment 1: Baseline ---
    baseline_params = {
        'objective': 'binary:logistic',
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    train_model(X_train, X_test, y_train, y_test, params=baseline_params, run_name="Baseline")

    # --- Experiment 2: Deeper Trees ---
    deeper_trees_params = baseline_params.copy()
    deeper_trees_params['max_depth'] = 8
    train_model(X_train, X_test, y_train, y_test, params=deeper_trees_params, run_name="Deeper Trees (max_depth=8)")

    # --- Run Experiment 3: Handle Class Imbalance ---
    # Since only ~3% of customers repeat, we need to handle the class imbalance.
    # 'scale_pos_weight' is a common technique for this.
    imbalance_params = baseline_params.copy()
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    imbalance_params['scale_pos_weight'] = scale_pos_weight
    train_model(X_train, X_test, y_train, y_test, params=imbalance_params, run_name="Handle Class Imbalance")

    print("\nAll experiments complete. Check the MLflow UI!")

if __name__ == "__main__":
    main()