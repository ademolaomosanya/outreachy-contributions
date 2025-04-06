from tdc.single_pred import ADME
import pandas as pd
import numpy as np
import os
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from time import time
import pickle
import warnings
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_advanced_features(smiles):
    """
    Generate comprehensive molecular features including advanced descriptors.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return None
        
        features = []
        feature_names = []
        
        try:
            # 1. Morgan Fingerprints (ECFP4) using MorganGenerator
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_array = np.zeros((2048,))
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            features.extend(fp_array)
            feature_names.extend([f'fp_{i}' for i in range(2048)])
        except Exception as e:
            logger.error(f"Error generating Morgan fingerprints: {str(e)}")
            return None
        
        try:
            # 2. MACCS Keys
            maccs = MACCSkeys.GenMACCSKeys(mol)
            maccs_array = np.zeros((167,))
            DataStructs.ConvertToNumpyArray(maccs, maccs_array)
            features.extend(maccs_array)
            feature_names.extend([f'maccs_{i}' for i in range(167)])
        except Exception as e:
            logger.error(f"Error generating MACCS keys: {str(e)}")
            return None
        
        try:
            # 3. Advanced Descriptors
            advanced_descriptors = {
                'TPSA': Descriptors.TPSA(mol),
                'MolLogP': Descriptors.MolLogP(mol),
                'MolWt': Descriptors.MolWt(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'FractionCSP3': Descriptors.FractionCSP3(mol),
                'RingCount': Descriptors.RingCount(mol),
                'QED': Descriptors.qed(mol),
                'BalabanJ': Descriptors.BalabanJ(mol),
                'BertzCT': Descriptors.BertzCT(mol)
            }
            
            features.extend(list(advanced_descriptors.values()))
            feature_names.extend(list(advanced_descriptors.keys()))
        except Exception as e:
            logger.error(f"Error calculating advanced descriptors: {str(e)}")
            return None
        
        try:
            # 4. Custom Absorption-Specific Features
            absorption_features = {
                'PolarSurfaceArea': Descriptors.TPSA(mol),
                'HBD_HBA_Ratio': Descriptors.NumHDonors(mol) / (Descriptors.NumHAcceptors(mol) + 1e-6),
                'LogP_MW_Ratio': Descriptors.MolLogP(mol) / (Descriptors.MolWt(mol) + 1e-6),
                'RotatableBonds_MW_Ratio': Descriptors.NumRotatableBonds(mol) / (Descriptors.MolWt(mol) + 1e-6)
            }
            
            features.extend(list(absorption_features.values()))
            feature_names.extend(list(absorption_features.keys()))
        except Exception as e:
            logger.error(f"Error calculating absorption features: {str(e)}")
            return None
        
        return np.array(features), feature_names
    except Exception as e:
        logger.error(f"Unexpected error in feature generation: {str(e)}")
        return None

def train_and_evaluate():
    """
    Train and evaluate the model with improved features and evaluation.
    """
    try:
        logger.info("Loading dataset...")
        # Load dataset
        data = ADME(name='HIA_Hou')
        df = data.get_data()
        
        # Generate features
        logger.info("Generating features...")
        features_list = []
        feature_names = None
        valid_indices = []
        failed_count = 0
        
        for idx, smiles in tqdm(enumerate(df['Drug']), total=len(df), desc="Generating features"):
            result = generate_advanced_features(smiles)
            if result is not None:
                feat, names = result
                features_list.append(feat)
                if feature_names is None:
                    feature_names = names
                valid_indices.append(idx)
            else:
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(f"Failed to generate features for {failed_count} molecules")
        
        if not features_list:
            raise ValueError("No valid features were generated")
            
        features = np.array(features_list)
        labels = df['Y'].values[valid_indices]
        
        logger.info(f"Generated features for {len(features)} molecules")
        logger.info(f"Feature dimension: {features.shape[1]}")
        
        # Split data with stratification
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create pipeline with SMOTE and feature selection
        logger.info("Setting up model pipeline...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('feature_selection', SelectFromModel(
                xgb.XGBClassifier(random_state=42),
                threshold='median'
            )),
            ('classifier', xgb.XGBClassifier(
                random_state=42,
                early_stopping_rounds=10,
                eval_metric='auc'
            ))
        ])
        
        # Hyperparameter grid
        param_grid = {
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__n_estimators': [100, 200],
            'classifier__subsample': [0.8, 1.0],
            'classifier__colsample_bytree': [0.8, 1.0]
        }
        
        # Perform grid search with cross-validation
        logger.info("Starting grid search...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        logger.info("Training model...")
        grid_search.fit(
            X_train, 
            y_train,
            classifier__eval_set=[(X_test, y_test)]
        )
        
        # Get best model and predictions
        logger.info("Making predictions...")
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_scores = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate comprehensive metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_scores),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        }
        
        # Log metrics
        logger.info("Model Performance:")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                logger.info(f"{metric}:")
                for param, val in value.items():
                    logger.info(f"    {param}: {val}")
            else:
                logger.info(f"{metric}: {value}")
        
        # Save model artifacts
        logger.info("Saving model artifacts...")
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/best_model.joblib')
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig('models/confusion_matrix.png')
        
        return metrics, feature_names
        
    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        metrics, feature_names = train_and_evaluate()
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            if metric not in ['best_params', 'cv_score']:
                print(f"{metric}: {value:.4f}")
        print("\nBest Parameters:", metrics['best_params'])
        print("Cross-validation Score:", metrics['cv_score'])
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1) 