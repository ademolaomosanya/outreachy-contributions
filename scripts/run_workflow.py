import os
import numpy as np
import pandas as pd
from tdc.single_pred import ADME
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def setup_directories():
    """Create necessary directories"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def generate_molecular_features(smiles):
    """Generate molecular features from SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate Morgan fingerprint
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_FP_RADIUS, nBits=MORGAN_FP_BITS)
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
        
        # Combine features
        features = np.concatenate([fp_array, descriptors])
        feature_names = [f'fp_{i}' for i in range(MORGAN_FP_BITS)] + descriptor_names
        
        return features, feature_names
    except:
        return None

def train_and_evaluate():
    """Train and evaluate the model"""
    print("1. Loading dataset...")
    data = ADME(name='HIA_Hou')
    df = data.get_data()
    
    print("2. Generating features...")
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
    
    print("3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED,
        stratify=labels if STRATIFY else None
    )
    
    print("4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("5. Training model...")
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train_scaled, y_train)
    
    print("6. Making predictions...")
    y_pred = model.predict(X_test_scaled)
    y_scores = model.predict_proba(X_test_scaled)[:, 1]
    
    print("7. Calculating metrics...")
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_scores)
    }
    
    print("8. Saving data and models...")
    # Save visualization data
    with open(os.path.join(DATA_DIR, 'visualization_data.pkl'), 'wb') as f:
        pickle.dump({
            'y_test': y_test,
            'y_pred': y_pred,
            'y_scores': y_scores,
            'feature_names': feature_names,
            'metrics': metrics
        }, f)
    
    # Save model and scaler
    joblib.dump(model, os.path.join(MODELS_DIR, 'hia_model.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'feature_scaler.joblib'))
    
    return metrics

def plot_metrics(metrics):
    """Plot performance metrics"""
    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    values = [metrics['accuracy'], metrics['precision'], 
             metrics['recall'], metrics['f1'], metrics['roc_auc']]
    
    bars = plt.bar(metrics_names, values, color='skyblue')
    plt.ylim(0.9, 1.0)
    plt.title('Model Performance Metrics', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'performance_metrics.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low Absorption', 'High Absorption'],
                yticklabels=['Low Absorption', 'High Absorption'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_scores):
    """Plot ROC curve"""
    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'roc_curve.png'))
    plt.close()

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.bar(range(20), importance[indices[:20]], color='skyblue')
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=45, ha='right')
    plt.title('Top 20 Most Important Features', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'feature_importance.png'))
    plt.close()

def generate_visualizations():
    """Generate all visualizations"""
    print("9. Generating visualizations...")
    
    # Load data
    with open(os.path.join(DATA_DIR, 'visualization_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    # Load model
    model = joblib.load(os.path.join(MODELS_DIR, 'hia_model.joblib'))
    
    # Generate plots
    plot_metrics(data['metrics'])
    plot_confusion_matrix(data['y_test'], data['y_pred'])
    plot_roc_curve(data['y_test'], data['y_scores'])
    plot_feature_importance(model, data['feature_names'])
    
    print("Visualizations saved in 'data' directory:")
    print("- Performance metrics: data/performance_metrics.png")
    print("- Confusion matrix: data/confusion_matrix.png")
    print("- ROC curve: data/roc_curve.png")
    print("- Feature importance: data/feature_importance.png")

def main():
    """Main workflow"""
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Setup directories
    setup_directories()
    
    # Train and evaluate model
    metrics = train_and_evaluate()
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate visualizations
    generate_visualizations()

if __name__ == "__main__":
    main() 