from tdc.single_pred import ADME
import pandas as pd
import numpy as np
import os
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from time import time
import pickle

def generate_molecular_features(smiles):
    """
    Generate molecular fingerprints and descriptors for a SMILES string.
    Returns None if the SMILES string is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate Morgan fingerprint (ECFP4)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fingerprint, fp_array)
        
        # Calculate molecular descriptors
        descriptors = []
        descriptor_names = []
        for name, function in Descriptors._descList:
            try:
                value = function(mol)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    descriptors.append(value)
                    descriptor_names.append(name)
            except:
                continue
        
        # Combine fingerprints and descriptors
        features = np.concatenate([fp_array, descriptors])
        feature_names = [f'fp_{i}' for i in range(1024)] + descriptor_names
        
        return features, feature_names
    except:
        return None

def train_and_evaluate():
    """
    Train and evaluate the XGBoost model on molecular features.
    """
    # Load dataset
    data = ADME(name='HIA_Hou')
    df = data.get_data()
    
    # Generate features
    features = []
    feature_names = []
    valid_indices = []
    
    for idx, smiles in enumerate(df['Drug']):
        result = generate_molecular_features(smiles)
        if result is not None:
            features.append(result[0])
            if not feature_names:  # Only add feature names once
                feature_names = result[1]
            valid_indices.append(idx)
    
    features = np.array(features)
    labels = df['Y'].values[valid_indices]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_scores = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_scores)
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    # Save data for visualization
    os.makedirs('data', exist_ok=True)
    with open('data/visualization_data.pkl', 'wb') as f:
        pickle.dump({
            'y_test': y_test,
            'y_pred': y_pred,
            'y_scores': y_scores,
            'feature_names': feature_names,
            'metrics': metrics
        }, f)
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/hia_model.joblib')
    joblib.dump(scaler, 'models/feature_scaler.joblib')
    
    return metrics

if __name__ == "__main__":
    train_and_evaluate() 