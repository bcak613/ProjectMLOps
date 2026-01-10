from fastapi import FastAPI
import mlflow.xgboost

app = FastAPI(title="Churn Prediction API")

@app.get("/")
def read_root():
    return {"message": "Welcome to Churn Prediction API"}

@app.post("/predict")
def predict(customer_id: str):
    # TODO: Get features from Feast
    # TODO: Predict using loaded model
    return {"customer_id": customer_id, "churn_prediction": 0.5}
