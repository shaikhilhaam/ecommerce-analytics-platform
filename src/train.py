# src/train.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import lightgbm as lgb
import os

def main():
    """
    Main function to run cross-validation training with LightGBM
    and log the results to MLflow.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow_data/mlflow.db")
    mlflow.set_experiment("Repeat Purchase Propensity")

    # Load the dataset with advanced features
    data_path = os.path.join('data', 'processed', 'propensity_dataset_advanced.csv')
    modeling_df = pd.read_csv(data_path)

    X = modeling_df.drop(columns=['customer_unique_id', 'is_repeat'])
    y = modeling_df['is_repeat']

    # Impute missing values (as before)
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # --- Stratified K-Fold Cross-Validation ---
    # Stratification ensures each fold has the same proportion of repeaters/non-repeaters
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Define model parameters (a strong baseline for LightGBM)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum()
    }
    
    fold_metrics = []

    # Start a parent MLflow run to group the cross-validation runs
    with mlflow.start_run(run_name=f"LGBM_CrossValidation_{n_splits}_folds") as parent_run:
        mlflow.log_params(params)
        
        for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
            # Start a nested run for each fold
            with mlflow.start_run(run_name=f"fold_{fold+1}", nested=True) as child_run:
                print(f"--- Starting Fold {fold+1}/{n_splits} ---")
                
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          eval_metric='f1',
                          callbacks=[lgb.early_stopping(100, verbose=False)])
                
                y_pred = model.predict(X_val)
                
                f1 = f1_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                
                fold_metrics.append({'f1': f1, 'precision': precision, 'recall': recall})
                
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

        # --- Log average metrics to the parent run ---
        avg_f1 = np.mean([m['f1'] for m in fold_metrics])
        avg_precision = np.mean([m['precision'] for m in fold_metrics])
        avg_recall = np.mean([m['recall'] for m in fold_metrics])
        
        print("\n--- Cross-Validation Summary ---")
        print(f"Average F1-Score: {avg_f1:.3f}")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        
        mlflow.log_metric("avg_f1_score", avg_f1)
        mlflow.log_metric("avg_precision", avg_precision)
        mlflow.log_metric("avg_recall", avg_recall)

        # --- Train Final Model on All Data ---
        print("\n--- Training Final Model on All Data ---")
        final_model = lgb.LGBMClassifier(**params)
        final_model.fit(X, y) # Train on the entire dataset
        
        # Log the final model
        mlflow.lightgbm.log_model(
            lgb_model=final_model,
            name="lightgbm-propensity-model-final",
            input_example=X.head()
        )
        print("Final model logged to MLflow.")

if __name__ == "__main__":
    main()