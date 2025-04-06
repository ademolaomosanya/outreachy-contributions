"""
Configuration file for the HIA prediction project.
Contains all random seeds and parameters for reproducibility.
"""

# Random seeds
RANDOM_SEED = 42

# Dataset parameters
TEST_SIZE = 0.2
STRATIFY = True

# Feature generation parameters
MORGAN_FP_RADIUS = 2
MORGAN_FP_BITS = 2048
ATOM_PAIR_FP_BITS = 2048
TORSION_FP_BITS = 2048

# Model parameters
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 1,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'n_estimators': 100,
    'random_state': RANDOM_SEED,
    'scale_pos_weight': 0.155  # Based on class imbalance
}

# Cross-validation parameters
CV_FOLDS = 5
CV_SCORING = 'f1'

# Visualization parameters
PLOT_STYLE = 'seaborn'
PLOT_FIGSIZE = (10, 6)
PLOT_DPI = 300

# File paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
VISUALIZATIONS_DIR = 'data' 