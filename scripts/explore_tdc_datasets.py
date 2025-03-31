from tdc.benchmark_group import admet_group
from tdc.single_pred import ADME
import pandas as pd

def explore_tdc_datasets():
    """
    Explore available datasets in TDC and print their information.
    """
    print("Available ADMET Datasets in TDC:\n")
    
    # Initialize ADMET group
    group = admet_group(path='data/')
    
    # Get available datasets
    datasets = group.dataset_names
    
    # Print datasets in a formatted way
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")
    
    print("\nTotal number of datasets:", len(datasets))
    
    # Print more information about a specific dataset (HIA)
    print("\nDetailed information about hia_hou dataset:")
    data = ADME(name='HIA_Hou')
    df = data.get_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

if __name__ == "__main__":
    explore_tdc_datasets() 