import mlflow
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import mlflow.xgboost
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Set environment variables immediately
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["AWS_REGION"] = "us-east-1"

def evaluate_model(run_id, test_data_path):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print(f"Loading model from run {run_id}...")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)
    
    print(f"Loading test data from {test_data_path}...")
    df = pd.read_parquet(test_data_path)
    
    X_test = df.drop(columns=['Churn', 'customer_id', 'event_timestamp'])
    y_test = df['Churn']
    
    print("Evaluating model...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"Test Accuracy: {acc}")
    print(classification_report(y_test, preds))
    
    # Log metric to the existing run
    print(f"Logging test_accuracy to run {run_id}...")
    client = mlflow.tracking.MlflowClient()
    client.log_metric(run_id, "test_accuracy", acc)
    
    return acc

def promote_model(run_id, accuracy, threshold=0.7):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    if accuracy >= threshold:
        print(f"Model accuracy {accuracy} meets threshold {threshold}. Promoting to Staging.")
        client = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{run_id}/model"
        name = "churn-prediction-model"
        
        # Register model if not exists
        try:
            client.create_registered_model(name)
        except Exception:
            pass
            
        result = client.create_model_version(
            name=name,
            source=model_uri,
            run_id=run_id
        )
        
        # Use Model Aliases (modern approach) instead of Stages (deprecated)
        client.set_registered_model_alias(
            name=name,
            alias="staging",
            version=result.version
        )
        print(f"Model version {result.version} assigned alias '@staging'.")
    else:
        print(f"Model accuracy {accuracy} is below threshold {threshold}. Not promoting.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    accuracy = evaluate_model(args.run_id, args.data)
    promote_model(args.run_id, accuracy)

if __name__ == "__main__":
    main()



