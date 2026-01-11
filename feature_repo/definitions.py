from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# Define an entity for the customer
customer = Entity(name="customer_id", value_type=ValueType.INT64, description="customer identifier")

# Define the file source for the offline store
churn_source = FileSource(
    path=r"../data/processed/train_churn.parquet",
    timestamp_field="event_timestamp",
)

# Define the feature view
churn_features = FeatureView(
    name="churn_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="Age", dtype=Float32),
        Field(name="Gender", dtype=Int64),
        Field(name="Tenure", dtype=Float32),
        Field(name="Usage Frequency", dtype=Float32),
        Field(name="Support Calls", dtype=Float32),
        Field(name="Payment Delay", dtype=Float32),
        Field(name="Subscription Type", dtype=Int64),
        Field(name="Contract Length", dtype=Int64),
        Field(name="Total Spend", dtype=Float32),
        Field(name="Last Interaction", dtype=Float32),
    ],
    online=True,
    source=churn_source,
    tags={"team": "churn_prediction"},
)
