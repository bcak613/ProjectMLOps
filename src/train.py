import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import argparse
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(data_path, params):
    print(f"Loading training data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Separate features and target
    X = df.drop(columns=['Churn', 'customer_id', 'event_timestamp'])
    y = df['Churn']
    
    # Train test split is already done in processing, this script expects 'train_churn.parquet'
    # For simplicity, we train on the input file.
    # In a real pipeline, we might pass train and validation paths separately.
    
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    
    # Log metrics (training metrics)
    train_preds = model.predict(X)
    train_proba = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, train_preds)
    f1 = f1_score(y, train_preds)
    roc_auc = roc_auc_score(y, train_proba)
    
    mlflow.log_metric("train_accuracy", acc)
    mlflow.log_metric("train_f1", f1)
    mlflow.log_metric("train_roc_auc", roc_auc)
    
    # Log Confusion Matrix
    cm = confusion_matrix(y, train_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.titile('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Log Model
    mlflow.xgboost.log_model(model, "model")
    print("Model trained and logged to MLflow.")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Churn Prediction Model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (parquet)")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("churn-prediction")

    params = {
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        train_model(args.data, params)


