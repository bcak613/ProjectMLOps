import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import argparse
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import os

# Set environment variables immediately
print("Configuring MinIO environment variables at top of script...")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["AWS_REGION"] = "us-east-1"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Suppress MLflow requirements warning
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)

def train_model(df, params):
    # Separate features and target
    X = df.drop(columns=['Churn', 'customer_id', 'event_timestamp'])
    y = df['Churn']
    
    print("Training XGBoost model...")
    # Ensure autolog is disabled
    print("Disabling autolog in train_model...")
    # mlflow.xgboost.autolog(disable=True)
    
    print("Initializing XGBClassifier...")
    model = xgb.XGBClassifier(**params)
    print("Fitting model...")
    model.fit(X, y)
    print("Model fit completed.")
    
    # Log metrics (training metrics)
    train_preds = model.predict(X)
    train_proba = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, train_preds)
    f1 = f1_score(y, train_preds)
    roc_auc = roc_auc_score(y, train_proba)
    
    print("Logging metrics to MLflow...")
    mlflow.log_metric("train_accuracy", acc)
    mlflow.log_metric("train_f1", f1)
    mlflow.log_metric("train_roc_auc", roc_auc)
    
    # Log Confusion Matrix
    print("Logging confusion matrix artifact...")
    cm = confusion_matrix(y, train_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Churn Prediction Model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (parquet)")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading training data from {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"Data loaded successfully. Shape: {df.shape}")

    print(f"Setting tracking URI to http://127.0.0.1:5000...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.autolog(disable=True)
    
    print(f"Setting experiment to 'churn-prediction-new'...")
    mlflow.set_experiment("churn-prediction-new")

    params = {
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "eval_metric": "logloss"
    }

    print("Starting MLflow run...")
    with mlflow.start_run():
        print("Logging parameters...")
        mlflow.log_params(params)
        
        # Train and log model manually
        model = train_model(df, params)
        
        # Log model artifact manually
        print("Logging model to MLflow...")
        mlflow.xgboost.log_model(model, "model")
        print("Model logged successfully.")

if __name__ == "__main__":
    main()


