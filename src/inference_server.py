from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from feast import FeatureStore
import os
import uvicorn
import json

import shap

app = FastAPI(title="Churn Prediction Inference Server")

# Configuration
FEATURE_REPO_PATH = "feature_repo"
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Set environment variables for MinIO access
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["AWS_REGION"] = "us-east-1"

# Global variables to hold model, store, and explainer
model = None
store = None
explainer = None

@app.on_event("startup")
def load_resources():
    global model, store, explainer
    print("Loading resources...")
    
    # 1. Load Feast Store
    store = FeatureStore(repo_path=FEATURE_REPO_PATH)
    
    # 2. Load Model from MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        experiment = mlflow.get_experiment_by_name("churn-prediction-new")
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if not runs.empty:
                run_id = runs.iloc[0].run_id
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.xgboost.load_model(model_uri)
                print(f"Model loaded from run: {run_id}")
                
                # 3. Initialize SHAP Explainer
                # For XGBoost, we can use TreeExplainer
                explainer = shap.TreeExplainer(model)
                print("SHAP Explainer initialized.")
    except Exception as e:
        print(f"Error loading model or explainer: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None, 
        "feast_connected": store is not None,
        "explainer_ready": explainer is not None
    }

@app.get("/predict/{customer_id}")
def predict(customer_id: int):
    if model is None or store is None:
        raise HTTPException(status_code=503, detail="Model or Feature Store not initialized")
    
    try:
        # 1. Fetch Features from Feast
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
        
        # Check if data exists
        if feature_vector.get('Age') is None and feature_vector.get('churn_features:Age') is None:
            raise HTTPException(status_code=404, detail="Customer not found")
            
        # 2. Prepare Data
        inference_features = {}
        for k, v in feature_vector.items():
            clean_key = k.split(":")[-1]
            inference_features[clean_key] = v[0]
            
        expected_cols = [
            "Age", "Gender", "Tenure", "Usage Frequency", "Support Calls", 
            "Payment Delay", "Subscription Type", "Contract Length", 
            "Total Spend", "Last Interaction"
        ]
        
        df = pd.DataFrame([inference_features])
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]
        
        # 3. Predict
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)[0][1]
        else:
            dtest = xgb.DMatrix(df)
            prob = model.predict(dtest)[0]
            
        # 4. Calculate SHAP values
        shap_explanation = {}
        if explainer is not None:
            shap_values = explainer.shap_values(df)
            # shap_values is an array of shape (1, n_features)
            # We convert it to a dictionary mapping feature name to SHAP value
            for i, col in enumerate(expected_cols):
                shap_explanation[col] = float(shap_values[0][i])
            
        return {
            "customer_id": customer_id,
            "features": inference_features,
            "probability": float(prob),
            "is_churn": bool(prob > 0.5),
            "shap_values": shap_explanation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
