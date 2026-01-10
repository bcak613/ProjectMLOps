import mlflow
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import mlflow.xgboost

def evaluate_model(run_id, test_data_path):
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
    
    # Log evaluation metrics to the same run (or a new one if preferred, but usually we attach to the training run or a child run)
    # Since run_id is passed, we can assume we might want to log externally or just print for CI/CD to capture.
    # For this level, we return the metrics.
    return acc

def promote_model(run_id, accuracy, threshold=0.7):
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
        
        client.transition_model_version_stage(
            name=name,
            version=result.version,
            stage="Staging"
        )
        print(f"Model version {result.version} promoted to Staging.")
    else:
        print(f"Model accuracy {accuracy} is below threshold {threshold}. Not promoting.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    accuracy = evaluate_model(args.run_id, args.data)
    promote_model(args.run_id, accuracy)



