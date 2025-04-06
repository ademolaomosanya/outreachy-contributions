import os
import pandas as pd
import requests
from pathlib import Path

def download_and_save_dataset():
    """Download the HIA_Hou dataset and save it locally."""
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # URL to the dataset
    url = 'https://raw.githubusercontent.com/mims-harvard/TDC/main/data/adme/HIA_Hou.csv'
    
    try:
        # Download the dataset
        print("Downloading HIA_Hou dataset...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the dataset
        dataset_path = data_dir / 'hia_hou_dataset.csv'
        with open(dataset_path, 'w') as f:
            f.write(response.text)
        
        # Read the dataset to generate statistics
        df = pd.read_csv(dataset_path)
        
        # Generate dataset information
        info = []
        info.append("Dataset Statistics:")
        info.append(f"Number of samples: {len(df)}")
        info.append(f"Number of features: {len(df.columns)}")
        info.append("\nClass Distribution:")
        class_dist = df['Y'].value_counts()
        info.append(f"Class 0: {class_dist.get(0, 0)}")
        info.append(f"Class 1: {class_dist.get(1, 0)}")
        
        info.append("\nSMILES Statistics:")
        smiles_lengths = df['Drug'].str.len()
        info.append(f"Average SMILES length: {smiles_lengths.mean():.2f}")
        info.append(f"Min SMILES length: {smiles_lengths.min()}")
        info.append(f"Max SMILES length: {smiles_lengths.max()}")
        
        info.append("\nSample Entries:")
        info.extend(df.head().to_string().split('\n'))
        
        # Save dataset information
        info_path = data_dir / 'hia_hou_info.txt'
        with open(info_path, 'w') as f:
            f.write('\n'.join(info))
            
        print(f"Dataset downloaded successfully to {dataset_path}")
        print(f"Dataset information saved to {info_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        
if __name__ == '__main__':
    download_and_save_dataset() 