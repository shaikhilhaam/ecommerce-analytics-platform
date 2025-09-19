# src/train.py
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
import os
import optuna # <-- Import Optuna

def objective(trial, X_train, y_train, X_test, y_test):
    """
    The objective function for Optuna to optimize.
    A 'trial' is a single run with a specific set of hyperparameters.
    """
    # Define the hyperparameter search space
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        # We keep our best imbalance handling technique constant
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum()
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # We want to maximize the F1 score, as it balances precision and recall
    f1 = f1_score(y_test, y_pred)
    return f1

def main():
    """Main function to run hyperparameter tuning with Optuna."""
    mlflow.set_tracking_uri("sqlite:///mlflow_data/mlflow.db")
    mlflow.set_experiment("Repeat Purchase Propensity")

    data_path = os.path.join('data', 'processed', 'propensity_dataset.csv')
    modeling_df = pd.read_csv(data_path)

    X = modeling_df.drop(columns=['customer_unique_id', 'is_repeat'])
    y = modeling_df['is_repeat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute missing values (as before)
    for col in X_train.columns:
        if X_train[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
            
    # --- Run Optuna Study ---
    study = optuna.create_study(direction='maximize')
    # Pass the data to the objective function using a lambda
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=20)

    print("Optuna study finished.")
    print("Best trial F1-score:", study.best_value)
    print("Best parameters found: ", study.best_params)

    # --- Train and log the final best model using the best params ---
    best_params = study.best_params
    best_params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum() # Re-add our constant param
    
    with mlflow.start_run(run_name="Optimized Model (Optuna)") as run:
        print(f"--- Starting Final Model Training with Best Params ---")
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        
        # Log everything for the best model
        mlflow.log_params(best_params)
        # You can re-calculate and log all metrics here if you want
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("f1_score", f1)
        
        input_example = X_train.head()
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost-propensity-model-optimized",
            input_example=input_example
        )
        print("Final optimized model logged to MLflow.")

if __name__ == "__main__":
    main()