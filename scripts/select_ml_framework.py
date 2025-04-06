from tdc.single_pred import ADME
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

def analyze_ml_frameworks():
    """
    Analyze different ML frameworks and select the most appropriate one for the HIA prediction task.
    Tests multiple classifiers on a simple feature set to compare performance.
    """
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("1. Loading HIA_Hou dataset...")
    data = ADME(name='HIA_Hou')
    df = data.get_data()
    
    # Step 1: Simple feature extraction (for comparison purposes only)
    print("\n2. Extracting simple features for framework comparison...")
    # For this comparison, we'll just use string length and basic character counts
    # In the real model, we'll use proper molecular fingerprints
    df['smiles_length'] = df['Drug'].str.len()
    df['carbon_count'] = df['Drug'].str.count('C')
    df['oxygen_count'] = df['Drug'].str.count('O')
    df['nitrogen_count'] = df['Drug'].str.count('N')
    df['ring_count'] = df['Drug'].str.count('1') + df['Drug'].str.count('2') + df['Drug'].str.count('3')
    
    # Simple features for quick test
    features = ['smiles_length', 'carbon_count', 'oxygen_count', 'nitrogen_count', 'ring_count']
    X = df[features]
    y = df['Y']
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Step 3: Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale entire dataset for cross-validation
    X_scaled = scaler.fit_transform(X)
    
    # Step 4: Define models to test
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear'),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'SVM': SVC(class_weight='balanced', probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), 
                                     random_state=42)
    }
    
    # Step 5: Compare models
    print("\n3. Comparing ML frameworks...")
    results = {}
    cv_results = {}
    
    for name, model in models.items():
        start_time = time()
        
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_scaled)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Track training time
        train_time = time() - start_time
        
        # Store results
        results[name] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'training_time': train_time
        }
        
        # Cross-validation for robustness
        cv_start = time()
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
        cv_results[name] = {
            'mean_cv_f1': cv_scores.mean(),
            'std_cv_f1': cv_scores.std(),
            'cv_time': time() - cv_start
        }
        
        print(f"- {name} - F1: {f1:.4f}, AUC: {auc:.4f}, Train Time: {train_time:.4f}s")
    
    # Step 6: Create comparison plot
    plt.figure(figsize=(10, 6))
    models_names = list(results.keys())
    f1_scores = [results[model]['f1_score'] for model in models_names]
    auc_scores = [results[model]['auc'] for model in models_names]
    
    x = np.arange(len(models_names))
    width = 0.35
    
    plt.bar(x - width/2, f1_scores, width, label='F1 Score')
    plt.bar(x + width/2, auc_scores, width, label='AUC')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/model_comparison.png')
    
    # Step 7: Save analysis to file
    with open('data/ml_framework_selection.txt', 'w') as f:
        f.write("ML Framework Selection for HIA Prediction\n")
        f.write("=======================================\n\n")
        
        f.write("1. Models Compared\n")
        f.write("--------------\n")
        for name in models.keys():
            f.write(f"- {name}\n")
        f.write("\n")
        
        f.write("2. Performance Metrics\n")
        f.write("-------------------\n")
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            for metric, value in metrics.items():
                f.write(f"  - {metric}: {value:.4f}\n")
            f.write(f"  - CV F1 Score: {cv_results[name]['mean_cv_f1']:.4f} ± {cv_results[name]['std_cv_f1']:.4f}\n")
            f.write("\n")
        
        # Select best model based on F1 score (handles class imbalance well)
        best_model = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
        best_cv_model = max(cv_results.items(), key=lambda x: x[1]['mean_cv_f1'])[0]
        
        f.write("3. Framework Selection\n")
        f.write("-------------------\n")
        f.write(f"Best model by test F1 score: {best_model}\n")
        f.write(f"Best model by cross-validation: {best_cv_model}\n\n")
        
        f.write("4. Justification\n")
        f.write("-------------\n")
        f.write("Scikit-learn with XGBoost integration is selected as the primary ML framework for this project because:\n\n")
        f.write("a) Dataset Characteristics:\n")
        f.write("   - Small dataset size (578 samples) suits traditional ML algorithms\n")
        f.write("   - Class imbalance (86.51% vs 13.49%) can be handled by these methods\n")
        f.write("   - Binary classification is well-supported\n\n")
        
        f.write("b) Computational Efficiency:\n")
        f.write("   - Quick training times suitable for iterative development\n")
        f.write("   - Low memory requirements align with feasibility assessment\n")
        f.write("   - No GPU required for these models\n\n")
        
        f.write("c) Model Interpretability:\n")
        f.write("   - Feature importance analysis available\n")
        f.write("   - Well-established evaluation metrics\n")
        f.write("   - Easy pipeline integration with RDKit and Ersilia features\n\n")
        
        f.write("d) Integration Capabilities:\n")
        f.write("   - Seamless integration with Python data science ecosystem\n")
        f.write("   - Compatible with molecular featurization from RDKit\n")
        f.write("   - Works well with Ersilia Model Hub for feature extraction\n\n")
        
        f.write("e) Model Serialization:\n")
        f.write("   - Easy model saving/loading with pickle or joblib\n")
        f.write("   - Portable for deployment\n")
        
    print("\n4. Analysis complete! Results saved to 'data/ml_framework_selection.txt'")
    print(f"   Plot saved to 'data/model_comparison.png'")
    print(f"\n5. Selected framework: Scikit-learn with {best_model} integration")

if __name__ == "__main__":
    analyze_ml_frameworks() 