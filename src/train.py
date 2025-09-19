# src/train.py
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import os
from imblearn.over_sampling import SMOTE

def train_model(X_train, X_test, y_train, y_test, params, run_name):
    # This function remains the same
    with mlflow.start_run(run_name=run_name) as run:
        print(f"--- Starting MLflow Run: {run.info.run_name} ---")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Run '{run_name}': Accuracy={accuracy:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        input_example = X_train.head()
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost-propensity-model",
            input_example=input_example
        )

def main():
    """Main function to load data and run multiple training experiments."""
    mlflow.set_tracking_uri("sqlite:///mlflow_data/mlflow.db")
    mlflow.set_experiment("Repeat Purchase Propensity")

    data_path = os.path.join('data', 'processed', 'propensity_dataset.csv')
    modeling_df = pd.read_csv(data_path)

    X = modeling_df.drop(columns=['customer_unique_id', 'is_repeat'])
    y = modeling_df['is_repeat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- FIX: Impute any remaining NaNs before using SMOTE ---
    print("Imputing remaining missing values...")
    for col in X_train.columns:
        if X_train[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
    print("Imputation complete.")

    # --- Run Baseline Experiment (with class imbalance handling) ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    imbalance_params = {
        'objective': 'binary:logistic', 'n_estimators': 200, 'max_depth': 4,
        'learning_rate': 0.05, 'use_label_encoder': False, 'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight
    }
    train_model(X_train, X_test, y_train, y_test, params=imbalance_params, run_name="Baseline (with scale_pos_weight)")

    # --- Run NEW Experiment: SMOTE Resampling ---
    print("\nApplying SMOTE to the training data...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"Original training set shape: {y_train.value_counts().to_dict()}")
    print(f"SMOTE resampled training set shape: {y_train_smote.value_counts().to_dict()}")

    smote_params = {
        'objective': 'binary:logistic', 'n_estimators': 200, 'max_depth': 4,
        'learning_rate': 0.05, 'use_label_encoder': False, 'eval_metric': 'logloss'
    }
    train_model(X_train_smote, X_test, y_train_smote, y_test, params=smote_params, run_name="SMOTE Resampling")

    print("\nAll experiments complete. Check the MLflow UI!")

if __name__ == "__main__":
    main()