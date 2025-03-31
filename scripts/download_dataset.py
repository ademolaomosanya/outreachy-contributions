from tdc.single_pred import ADME
import pandas as pd
import os

def download_and_save_dataset():
    """
    Download the HIA_Hou dataset from TDC and save it in the data folder.
    Also saves a data info file with basic statistics.
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("1. Downloading HIA_Hou dataset...")
    data = ADME(name='HIA_Hou')
    df = data.get_data()
    
    # Save the main dataset
    output_file = os.path.join('data', 'hia_hou_dataset.csv')
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to: {output_file}")
    
    # Save data info
    info_file = os.path.join('data', 'hia_hou_info.txt')
    with open(info_file, 'w') as f:
        f.write("HIA_Hou Dataset Information\n")
        f.write("==========================\n\n")
        
        f.write("1. Basic Statistics\n")
        f.write("-----------------\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Features: {', '.join(df.columns)}\n")
        f.write(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB\n\n")
        
        f.write("2. Class Distribution\n")
        f.write("-------------------\n")
        class_dist = df['Y'].value_counts()
        f.write(f"Class 0 (Not Absorbed): {class_dist[0]} samples\n")
        f.write(f"Class 1 (Absorbed): {class_dist[1]} samples\n\n")
        
        f.write("3. SMILES Statistics\n")
        f.write("-----------------\n")
        f.write(f"Average SMILES length: {df['Drug'].str.len().mean():.1f} characters\n")
        f.write(f"Min SMILES length: {df['Drug'].str.len().min()} characters\n")
        f.write(f"Max SMILES length: {df['Drug'].str.len().max()} characters\n\n")
        
        f.write("4. Sample Entries\n")
        f.write("--------------\n")
        f.write("First 3 entries:\n")
        for _, row in df.head(3).iterrows():
            f.write(f"Drug ID: {row['Drug_ID']}\n")
            f.write(f"SMILES: {row['Drug']}\n")
            f.write(f"Label: {row['Y']}\n")
            f.write("-" * 50 + "\n")
    
    print(f"Dataset info saved to: {info_file}")
    
    # Print summary
    print("\nDownload Summary:")
    print(f"- Dataset shape: {df.shape}")
    print(f"- Features: {', '.join(df.columns)}")
    print(f"- Class distribution: {dict(class_dist)}")
    print(f"- Files saved in: {os.path.abspath('data')}")

if __name__ == "__main__":
    download_and_save_dataset() 