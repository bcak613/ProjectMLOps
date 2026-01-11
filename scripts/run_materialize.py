from feast import FeatureStore
from datetime import datetime
import pandas as pd
import warnings
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.DEBUG)

def run_materialize():
    store = FeatureStore(repo_path="feature_repo")
    
    print("Starting materialization...")
    # List feature views to verify registry
    print("Feature Views:", [fv.name for fv in store.list_feature_views()])

    from datetime import timezone
    
    # Materialize for the specific day of the data
    start_date = datetime(2026, 1, 11, tzinfo=timezone.utc)
    end_date = datetime(2026, 1, 12, tzinfo=timezone.utc)
    
    # Manual materialization workaround
    print("Starting manual materialization...")
    try:
        # Read the parquet file directly
        df = pd.read_parquet("data/processed/train_churn.parquet")
        print(f"Read {len(df)} rows from parquet.")
        
        # Ensure event_timestamp is timezone-aware UTC
        if df['event_timestamp'].dt.tz is None:
             df['event_timestamp'] = df['event_timestamp'].dt.tz_localize('UTC')
        else:
             df['event_timestamp'] = df['event_timestamp'].dt.tz_convert('UTC')

        # Write to online store in chunks
        chunk_size = 10000
        total_rows = len(df)
        print(f"Writing to online store in chunks of {chunk_size}...")
        
        for i in range(0, total_rows, chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(total_rows + chunk_size - 1)//chunk_size} (Rows {i} to {min(i+chunk_size, total_rows)})...")
            store.write_to_online_store(feature_view_name="churn_features", df=chunk)
            
        print("Manual materialization completed!")
        
    except Exception as e:
        print(f"Manual materialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_materialize()
