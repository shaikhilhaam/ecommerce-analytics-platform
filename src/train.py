# src/train.py
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import os
import numpy as np

def train_model(X_train, X_test, y_train, y_test, params, run_name):
    """Trains a single XGBoost model and logs it to MLflow under a specific run name."""
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Starting MLflow Run: {run.info.run_name}")

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Run '{run_name}': RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        input_example = X_train.head()
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost-ltv-model",
            input_example=input_example
        )

def main():
    """Main function to load data and run multiple training experiments."""
    mlflow.set_tracking_uri("sqlite:///mlflow_data/mlflow.db")
    mlflow.set_experiment("LTV Prediction")

    data_path = os.path.join('data', 'processed', 'modeling_dataset.csv')
    modeling_df = pd.read_csv(data_path)

    X = modeling_df.drop(columns=['customer_unique_id', 'ltv_90_days'])
    y = modeling_df['ltv_90_days']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Run Experiment 1: Our Baseline ---
    baseline_params = {
        'objective': 'reg:squarederror', 'n_estimators': 200, 'max_depth': 4,
        'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 42
    }
    train_model(X_train, X_test, y_train, y_test, params=baseline_params, run_name="Baseline")

    # --- Run Experiment 2: Deeper Trees ---
    deeper_trees_params = baseline_params.copy()
    deeper_trees_params['max_depth'] = 8
    train_model(X_train, X_test, y_train, y_test, params=deeper_trees_params, run_name="Deeper Trees (max_depth=8)")

    # --- Run Experiment 3: Higher Learning Rate ---
    higher_lr_params = baseline_params.copy()
    higher_lr_params['learning_rate'] = 0.1
    train_model(X_train, X_test, y_train, y_test, params=higher_lr_params, run_name="Higher Learning Rate (lr=0.1)")

    print("\nAll experiments complete. Check the MLflow UI!")

if __name__ == "__main__":
    main()