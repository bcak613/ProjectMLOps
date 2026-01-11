from feast import FeatureStore
import pandas as pd

def test_feature_retrieval():
    store = FeatureStore(repo_path="feature_repo")
    
    print("Fetching online features...")
    feature_vector = store.get_online_features(
        features=[
            "churn_features:Age",
            "churn_features:Total Spend",
            "churn_features:Tenure",
        ],
        entity_rows=[
            {"customer_id": 328860},
            {"customer_id": 179930},
            {"customer_id": 279562},
        ],
    ).to_dict()
    
    print("Features retrieved:")
    print(feature_vector)

if __name__ == "__main__":
    test_feature_retrieval()
