import os
import zipfile
import subprocess
import sys

def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print(f"Zip file not found: {zip_path}")
        return False
    
    print(f"Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")
    return True

def download_kaggle_dataset(dataset_id, download_path):
    print(f"Downloading Kaggle dataset: {dataset_id}...")
    os.makedirs(download_path, exist_ok=True)
    try:
        # Using subprocess to call kaggle cli
        subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset_id, 
            "-p", download_path, "--unzip"
        ], check=True)
        print(f"Downloaded and unzipped {dataset_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {dataset_id}: {e}")
        print("Make sure kaggle API is installed and kaggle.json is configured.")
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install it with 'pip install kaggle'")

if __name__ == "__main__":
    # 1. Handle local thesis dataset
    local_zip = "dehazing-dataset-thesis.zip"
    thesis_extract_path = "data/raw/thesis"
    if not os.path.exists(thesis_extract_path) or not os.listdir(thesis_extract_path):
        extract_zip(local_zip, thesis_extract_path)
    else:
        print("Thesis dataset already extracted.")

    # 2. Handle missing datasets from Kaggle
    kaggle_datasets = {
        "haze1k": "mohit-3430/haze1k-full",
        "rshaze": "hazel0/rshaze-dataset"
    }

    for name, ds_id in kaggle_datasets.items():
        ds_path = os.path.join("data/raw", name)
        if not os.path.exists(ds_path) or not os.listdir(ds_path):
            download_kaggle_dataset(ds_id, ds_path)
        else:
            print(f"Dataset {name} already exists.")
