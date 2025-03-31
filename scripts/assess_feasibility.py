from tdc.single_pred import ADME
import pandas as pd
import numpy as np
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import psutil
import os

def assess_computational_feasibility():
    """
    Assess the computational feasibility of working with the HIA_Hou dataset by analyzing:
    1. Dataset size and memory requirements
    2. Feature generation complexity
    3. Computational resources needed
    """
    print("Loading HIA_Hou dataset...")
    data = ADME(name='HIA_Hou')
    df = data.get_data()
    
    print("\nComputational Feasibility Assessment")
    print("==================================")
    
    # 1. Dataset Size Analysis
    print("\n1. Dataset Size:")
    print("--------------")
    memory_usage = df.memory_usage(deep=True).sum()
    print(f"Number of samples: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Memory usage: {memory_usage / 1024**2:.2f} MB")
    
    # 2. Feature Analysis
    print("\n2. Feature Analysis:")
    print("-----------------")
    print("Input features: SMILES strings")
    print(f"Average SMILES length: {df['Drug'].str.len().mean():.1f} characters")
    print(f"Max SMILES length: {df['Drug'].str.len().max()} characters")
    
    # 3. Molecular Descriptor Generation
    print("\n3. Molecular Descriptor Generation:")
    print("-------------------------------")
    # Calculate time for generating descriptors for a sample
    sample_smiles = df['Drug'].iloc[0]
    mol = Chem.MolFromSmiles(sample_smiles)
    n_descriptors = len(Descriptors._descList)
    print(f"Number of RDKit descriptors available: {n_descriptors}")
    
    # Generate Morgan fingerprints for first molecule
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    print(f"Morgan fingerprint length: {len(fingerprint.ToBitString())} bits")
    
    # 4. System Resources
    print("\n4. System Resources:")
    print("-----------------")
    memory = psutil.virtual_memory()
    print(f"Available system memory: {memory.available / 1024**3:.1f} GB")
    print(f"Total system memory: {memory.total / 1024**3:.1f} GB")
    print(f"CPU cores: {psutil.cpu_count()}")
    
    # 5. Storage Requirements
    print("\n5. Storage Requirements:")
    print("---------------------")
    estimated_descriptor_size = len(df) * n_descriptors * 8  # 8 bytes per float
    estimated_fingerprint_size = len(df) * 1024 // 8  # bits to bytes
    print(f"Estimated size for descriptors: {estimated_descriptor_size / 1024**2:.2f} MB")
    print(f"Estimated size for fingerprints: {estimated_fingerprint_size / 1024**2:.2f} MB")
    
    # Save feasibility assessment
    with open('data/feasibility_assessment.txt', 'w') as f:
        f.write("HIA_Hou Dataset Feasibility Assessment\n")
        f.write("====================================\n\n")
        
        f.write("1. Dataset Characteristics\n")
        f.write("------------------------\n")
        f.write(f"- Samples: {len(df)}\n")
        f.write(f"- Current memory usage: {memory_usage / 1024**2:.2f} MB\n")
        f.write(f"- Average SMILES length: {df['Drug'].str.len().mean():.1f} characters\n\n")
        
        f.write("2. Computational Requirements\n")
        f.write("--------------------------\n")
        f.write(f"- RDKit descriptors: {n_descriptors}\n")
        f.write(f"- Morgan fingerprint size: 1024 bits\n")
        f.write(f"- Estimated storage for features: {(estimated_descriptor_size + estimated_fingerprint_size) / 1024**2:.2f} MB\n\n")
        
        f.write("3. Feasibility Verdict\n")
        f.write("-------------------\n")
        f.write("The dataset is computationally feasible because:\n")
        f.write("1. Small dataset size (< 1000 samples)\n")
        f.write("2. Standard molecular descriptors and fingerprints\n")
        f.write("3. Minimal memory requirements\n")
        f.write("4. No need for distributed computing\n")
        f.write("\nRecommended Setup:\n")
        f.write("- Standard laptop/desktop computer\n")
        f.write("- 8GB+ RAM (16GB preferred)\n")
        f.write("- Standard CPU (no GPU required)\n")
        f.write("- Local storage (< 1GB needed)\n")

if __name__ == "__main__":
    assess_computational_feasibility() 