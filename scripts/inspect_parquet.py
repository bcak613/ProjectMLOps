import pandas as pd

df = pd.read_parquet("data/processed/train_churn.parquet")
print("Columns:", df.columns.tolist())
print("Timestamp Range:", df['event_timestamp'].min(), "to", df['event_timestamp'].max())
print("Timezone:", df['event_timestamp'].dt.tz)
