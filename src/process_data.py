import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    print("Cleaning data...")
    
    # Drop rows where Churn is NaN
    df.dropna(subset=['Churn'], inplace=True)
    
    # Basic preprocessing
    # The dataset uses 'Total Spend' instead of 'TotalCharges'
    if 'Total Spend' in df.columns:
        df['Total Spend'] = pd.to_numeric(df['Total Spend'], errors='coerce')
        df['Total Spend'].fillna(0, inplace=True)
    
    # Binary encoding for target
    # The dataset 'Churn' column might already be numeric (0/1) or 'Yes'/'No'
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Fill remaining NaNs with 0 or a placeholder
    df.fillna(0, inplace=True)
    
    # Rename CustomerID to customer_id for consistency with Feast
    if 'CustomerID' in df.columns:
        df.rename(columns={'CustomerID': 'customer_id'}, inplace=True)
    
    if 'customer_id' in df.columns:
        df['customer_id'] = df['customer_id'].astype(int)

    # Simple encoding for categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'customer_id': # Keep entity key as is
            df[col] = df[col].astype('category').cat.codes
            
    return df

def save_data(df, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{prefix}_churn.parquet")
    print(f"Saving {prefix} data to {file_path}")
    df.to_parquet(file_path, index=False)
    return file_path

def main():
    parser = argparse.ArgumentParser(description="Process data for Churn Prediction")
    parser.add_argument("--input", type=str, required=True, help="Path to raw data")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed data")
    args = parser.parse_args()

    df = load_data(args.input)
    df = clean_data(df)
    
    # Add timestamp for Feast if needed, using current time for simplicity (UTC)
    df['event_timestamp'] = pd.Timestamp.now(tz='UTC')
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    save_data(train_df, args.output, "train")
    save_data(test_df, args.output, "test")
    save_data(df, args.output, "data") # Full dataset for Feature Store

if __name__ == "__main__":
    main()
