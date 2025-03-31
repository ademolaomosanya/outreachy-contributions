from tdc.single_pred import ADME
import pandas as pd
import numpy as np

def analyze_dataset_background():
    """
    Analyze and document the background and endpoint information for the HIA_Hou dataset.
    """
    print("Loading HIA_Hou dataset...")
    data = ADME(name='HIA_Hou')
    df = data.get_data()
    
    print("\nDataset Background and Endpoint Analysis")
    print("=======================================")
    
    # 1. Dataset Overview
    print("\n1. Basic Information:")
    print("-------------------")
    print(f"Dataset Name: {data.name}")
    print(f"Total Compounds: {len(df)}")
    
    # 2. Molecular Representation
    print("\n2. Molecular Information:")
    print("----------------------")
    print("Input Format: SMILES (Simplified Molecular Input Line Entry System)")
    print("Example SMILES:", df['Drug'].iloc[0])
    print(f"Unique Molecules: {df['Drug'].nunique()}")
    
    # 3. Endpoint Information
    print("\n3. Endpoint Details:")
    print("-----------------")
    print("Property: Human Intestinal Absorption (HIA)")
    print("Type: Binary Classification")
    print("Labels:")
    print("  - 0: Not Absorbed")
    print("  - 1: Absorbed")
    
    # 4. Class Distribution
    value_counts = df['Y'].value_counts().sort_index()
    print("\n4. Class Distribution:")
    print("-------------------")
    for label, count in value_counts.items():
        print(f"Class {label}: {count} samples ({count/len(df)*100:.2f}%)")
    
    # Save detailed background information
    with open('data/dataset_background.txt', 'w') as f:
        f.write("HIA_Hou Dataset Background Information\n")
        f.write("====================================\n\n")
        
        f.write("1. Dataset Overview\n")
        f.write("-----------------\n")
        f.write("The HIA_Hou dataset is focused on Human Intestinal Absorption (HIA),\n")
        f.write("which is a crucial property in drug development and pharmaceutical research.\n\n")
        
        f.write("2. Scientific Background\n")
        f.write("---------------------\n")
        f.write("Human Intestinal Absorption (HIA) is the process by which drugs and other\n")
        f.write("compounds are absorbed from the gastrointestinal tract into the bloodstream.\n")
        f.write("This property is essential for oral drug delivery and bioavailability.\n\n")
        
        f.write("3. Endpoint Description\n")
        f.write("--------------------\n")
        f.write("The endpoint is binary:\n")
        f.write("- 0: Compound is NOT absorbed through the human intestine\n")
        f.write("- 1: Compound IS absorbed through the human intestine\n\n")
        
        f.write("4. Practical Significance\n")
        f.write("----------------------\n")
        f.write("Understanding HIA is crucial for:\n")
        f.write("- Drug development and screening\n")
        f.write("- Optimizing oral drug delivery\n")
        f.write("- Predicting drug bioavailability\n")
        f.write("- Reducing costs in drug development\n\n")
        
        f.write("5. Dataset Statistics\n")
        f.write("------------------\n")
        f.write(f"Total Compounds: {len(df)}\n")
        f.write(f"Unique Molecules: {df['Drug'].nunique()}\n")
        f.write("Class Distribution:\n")
        for label, count in value_counts.items():
            f.write(f"- Class {label}: {count} samples ({count/len(df)*100:.2f}%)\n")

if __name__ == "__main__":
    analyze_dataset_background() 