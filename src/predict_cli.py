import argparse
import json
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
import sys
import os
from feast import FeatureStore

# Set environment variables for MinIO access
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["AWS_REGION"] = "us-east-1"

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
FEATURE_REPO_PATH = "feature_repo"

def get_online_features(customer_id):
    try:
        store = FeatureStore(repo_path=FEATURE_REPO_PATH)
        feature_vector = store.get_online_features(
            features=[
                "churn_features:Age", "churn_features:Gender", "churn_features:Tenure",
                "churn_features:Usage Frequency", "churn_features:Support Calls",
                "churn_features:Payment Delay", "churn_features:Subscription Type",
                "churn_features:Contract Length", "churn_features:Total Spend",
                "churn_features:Last Interaction"
            ],
            entity_rows=[{"customer_id": customer_id}]
        ).to_dict()
        
        # Extract values
        features = {k: v[0] for k, v in feature_vector.items()}
        
        # Check if data exists
        # Feast might return keys with or without prefix depending on version/config
        # We check for 'Age' or 'churn_features:Age'
        if features.get('Age') is None and features.get('churn_features:Age') is None:
            return None
            
        return features
    except Exception as e:
        print(f"Feast Error: {e}", file=sys.stderr)
        return None

def load_model():
    try:
        # Fallback: Find latest run in experiment
        experiment = mlflow.get_experiment_by_name("churn-prediction-new")
        if experiment is None:
            return None
            
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            return None
            
        run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{run_id}/model"
        
        # Use xgboost loader
        return mlflow.xgboost.load_model(model_uri)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer_id", type=int, required=True, help="Customer ID")
    args = parser.parse_args()

    try:
        # 1. Fetch Features
        features = get_online_features(args.customer_id)
        if features is None:
            print(json.dumps({"error": "Customer not found"}))
            return

        # 2. Load Model
        model = load_model()
        if model is None:
            print(json.dumps({"error": "Model could not be loaded"}))
            return

        # 3. Prepare Data
        # Clean column names
        inference_features = {}
        for k, v in features.items():
            clean_key = k.split(":")[-1]
            inference_features[clean_key] = v
        
        # Ensure column order
        expected_cols = [
            "Age", "Gender", "Tenure", "Usage Frequency", "Support Calls", 
            "Payment Delay", "Subscription Type", "Contract Length", 
            "Total Spend", "Last Interaction"
        ]
        
        df = pd.DataFrame([inference_features])
        # Add missing cols with 0
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

        # 4. Predict
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)[0][1]
        else:
            dtest = xgb.DMatrix(df)
            prob = model.predict(dtest)[0]
            
        # 5. Output Result (Features + Prediction)
        result = {
            "features": inference_features,
            "probability": float(prob),
            "is_churn": bool(prob > 0.5)
        }
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        import traceback
        traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    main()
