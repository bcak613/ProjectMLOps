import os
import subprocess
import re

def download_kaggle_dataset():
    # Path to the data_url.txt file
    url_file = 'data_url.txt'
    
    if not os.path.exists(url_file):
        print(f"Error: {url_file} not found.")
        return

    # Read the URL from the file
    with open(url_file, 'r') as f:
        url = f.read().strip()

    if not url:
        print(f"Error: {url_file} is empty.")
        return

    # Extract the dataset identifier from the URL
    # Example URL: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset/data
    # Identifier: muhammadshahidazeem/customer-churn-dataset
    match = re.search(r'kaggle\.com/datasets/([^/]+/[^/]+)', url)
    if not match:
        print(f"Error: Could not parse Kaggle dataset identifier from URL: {url}")
        return

    dataset_id = match.group(1)
    print(f"Detected dataset: {dataset_id}")

    # Create the data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    # Download the dataset using Kaggle CLI
    print(f"Downloading dataset {dataset_id} to {data_dir}...")
    try:
        # Command: kaggle datasets download -d <identifier> -p data --unzip
        subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_id, '-p', data_dir, '--unzip'], check=True)
        print("Download and extraction completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during download: {e}")
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Please ensure the kaggle python package is installed and in your PATH.")

if __name__ == "__main__":
    download_kaggle_dataset()
