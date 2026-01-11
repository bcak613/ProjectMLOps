from feast import FeatureStore
import pandas as pd
from datetime import datetime

def test_offline_retrieval():
    store = FeatureStore(repo_path="feature_repo")
    
    entity_df = pd.DataFrame.from_dict({
        "customer_id": [328860, 179930, 279562],
        "event_timestamp": [
            datetime(2026, 1, 12),
            datetime(2026, 1, 12),
            datetime(2026, 1, 12),
        ]
    })
    
    print("Fetching historical features...")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "churn_features:Age",
            "churn_features:Total Spend",
            "churn_features:Tenure",
        ],
    ).to_df()
    
    print("Historical features retrieved:")
    print(training_df.head())

if __name__ == "__main__":
    test_offline_retrieval()
