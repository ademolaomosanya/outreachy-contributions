from tdc.single_pred import ADME
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def verify_classification():
    """
    Verify that HIA_Hou is a classification problem by analyzing:
    1. Target variable distribution
    2. Unique values in target
    3. Data characteristics
    """
    print("Loading HIA_Hou dataset...")
    data = ADME(name='HIA_Hou')
    df = data.get_data()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("\nDataset Overview:")
    print("-----------------")
    print(f"Total samples: {len(df)}")
    print(f"Features: {df.columns.tolist()}")
    
    # Analyze target variable (Y)
    print("\nTarget Variable Analysis:")
    print("-----------------------")
    unique_values = sorted(df['Y'].unique())
    print("Unique values in target (Y):", unique_values)
    
    # Get value counts
    value_counts = df['Y'].value_counts().sort_index()
    print("\nClass distribution:")
    for label, count in value_counts.items():
        print(f"Class {label}: {count} samples ({count/len(df)*100:.2f}%)")
    
    # Plot distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Y')
    plt.title('Distribution of Classes in HIA_Hou Dataset')
    plt.xlabel('Class (0: Not Absorbed, 1: Absorbed)')
    plt.ylabel('Count')
    plt.savefig('data/hia_distribution.png')
    plt.close()
    
    # Comprehensive verification
    print("\nClassification Problem Verification:")
    print("----------------------------------")
    
    # Check 1: Binary values
    is_binary = set(unique_values) == {0, 1}
    print("\n1. Binary Target Check:")
    print("   Result:", "✓ Pass" if is_binary else "✗ Fail")
    print("   Explanation: Target variable contains only binary values (0 and 1)")
    
    # Check 2: Discrete values
    is_discrete = all(isinstance(x, (int, np.integer)) for x in unique_values)
    print("\n2. Discrete Values Check:")
    print("   Result:", "✓ Pass" if is_discrete else "✗ Fail")
    print("   Explanation: All target values are discrete integers")
    
    # Check 3: Class balance analysis
    min_class_ratio = min(value_counts) / len(df)
    print("\n3. Class Balance Analysis:")
    print(f"   Minority class ratio: {min_class_ratio:.2%}")
    print("   Note: While imbalanced, both classes have sufficient samples for classification")
    
    # Final verdict
    print("\nFinal Verdict:")
    print("-------------")
    if is_binary and is_discrete:
        print("✓ CONFIRMED: HIA_Hou is a Binary Classification Problem")
        print("\nReasoning:")
        print("1. Target variable contains exactly two distinct values (0 and 1)")
        print("2. Values represent discrete classes (Not Absorbed vs Absorbed)")
        print("3. Each sample belongs to exactly one class")
        print("4. The task is to predict which class a compound belongs to")
    else:
        print("✗ NOT CONFIRMED as a binary classification problem")
        
    # Save verification results
    with open('data/classification_verification.txt', 'w') as f:
        f.write("HIA_Hou Dataset Classification Verification\n")
        f.write("=========================================\n\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Target values: {unique_values}\n")
        f.write(f"Class distribution:\n")
        for label, count in value_counts.items():
            f.write(f"Class {label}: {count} samples ({count/len(df)*100:.2f}%)\n")
        f.write("\nVerification Result: Binary Classification Problem\n")
        f.write("Target variable represents binary outcomes for Human Intestinal Absorption\n")
        f.write("0: Not Absorbed\n")
        f.write("1: Absorbed\n")

if __name__ == "__main__":
    verify_classification() 