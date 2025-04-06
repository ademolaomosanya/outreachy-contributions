import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import pickle
import os

def load_model_and_data():
    """Load the trained model and data"""
    model = joblib.load('models/hia_model.joblib')
    scaler = joblib.load('models/feature_scaler.joblib')
    
    # Load visualization data
    with open('data/visualization_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    return model, scaler, data

def plot_metrics(metrics):
    """Plot the performance metrics"""
    plt.figure(figsize=(10, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    values = [metrics['accuracy'], metrics['precision'], 
             metrics['recall'], metrics['f1'], metrics['roc_auc']]
    
    bars = plt.bar(metrics_names, values, color='skyblue')
    plt.ylim(0.9, 1.0)
    plt.title('Model Performance Metrics', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('data/performance_metrics.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """Plot the confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low Absorption', 'High Absorption'],
                yticklabels=['Low Absorption', 'High Absorption'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig('data/confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_scores):
    """Plot the ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
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
    plt.savefig('data/roc_curve.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(20), importance[indices[:20]], color='skyblue')
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=45, ha='right')
    plt.title('Top 20 Most Important Features', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')
    plt.close()

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load model and data
    model, scaler, data = load_model_and_data()
    
    # Generate plots
    plot_metrics(data['metrics'])
    plot_confusion_matrix(data['y_test'], data['y_pred'])
    plot_roc_curve(data['y_test'], data['y_scores'])
    plot_feature_importance(model, data['feature_names'])
    
    print("Visualizations have been generated and saved in the 'data' directory:")
    print("- Performance metrics: data/performance_metrics.png")
    print("- Confusion matrix: data/confusion_matrix.png")
    print("- ROC curve: data/roc_curve.png")
    print("- Feature importance: data/feature_importance.png")

if __name__ == "__main__":
    main() 