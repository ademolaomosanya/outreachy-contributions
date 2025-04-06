# Outreachy contribution period

This repository contains a basic folder structure to be used during the Outreachy contribution period. Please make a fork to start contributing, and detail progress in the open issue. 

## Project goals:
- Understand how to use and interact with the Ersilia Model Hub
- Demonstrate basic AI/ML knowledge
- Show your Python coding skills 
- Practice code documentation and end user documentation

## Structure overview
The template repository already has pre-defined folders. Please restrict your project to using them for easy review:
- data: folder where data needs to be stored once downloaded
- notebooks: jupyter notebooks
- scripts: python/bash scripts necessary to run the project
- models: folder with model checkpoints

## Project Description
The goal of this research is to create a machine learning model that can forecast a medicinal compound's Human Intestinal Absorption (HIA). The Therapeutics Data Commons (TDC) HIA_Hou dataset, which includes molecule structures and the associated absorption characteristics, will be used to train the model.

## Installation
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/your-username/outreachy-contributions.git
cd outreachy-contributions
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset Details

### HIA_Hou Dataset
- **Source**: Therapeutics Data Commons (TDC)
- **Task**: Binary classification of drug absorption
- **Size**: 578 compounds
- **Features**: 
  - Drug_ID: Compound identifier
  - Drug: SMILES representation of molecular structure
  - Y: Binary label (1: Absorbed, 0: Not absorbed)

- **The distribution of classes:**:
  - Absorbed (1): 86.51% of 500 compounds
  - Compounds not absorbed (0): 78 (13.49%)

### Format of Datasets
The following columns are included in the CSV file that contains the dataset:
- `Drug_ID`: Each compound's unique identifier
- `Drug`: The molecular structure is represented by the SMILES string
- `Y`: Label for binary absorption

## Model Building Process

### Overview
The model-building process involves several key steps:
1. Data loading and preprocessing
2. Feature generation from molecular structures
3. Model training and evaluation
4. Performance visualization

### Feature Generation
The model uses a comprehensive set of molecular features generated using the `MolecularFeaturizer` class. The featurization process is fully reproducible and includes the following feature types:

1. **Morgan Fingerprints (ECFP)**
   - Radius: 2
   - Number of bits: 2048
   - Captures local structural information and pharmacophore features

2. **MACCS Keys**
   - 167-bit binary vectors
   - Identifies specific functional groups and structural patterns

3. **Atom Pair Fingerprints**
   - 2048-bit binary vectors
   - Captures atom pair relationships and distances

4. **Topological Torsion Fingerprints**
   - 2048-bit binary vectors
   - Describes molecular connectivity and flexibility

5. **Molecular Descriptors**
   - Physicochemical properties (e.g., logP, molecular weight)
   - Electronic properties
   - Topological indices

6. **3D Molecular Descriptors**
   - Shape-based descriptors
   - Conformational properties
   - Generated using RDKit's 3D conformation generation

### Featurization Process
To generate features for your molecules:

1. **Setup the Environment**
   ```bash
   # Install required packages
   pip install -r requirements.txt
   ```

2. **Using the MolecularFeaturizer**
   ```python
   from scripts.featurize_molecules import MolecularFeaturizer
   
   # Initialize the featurizer
   featurizer = MolecularFeaturizer()
   
   # Generate features for a single molecule
   smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Example SMILES (Aspirin)
   features = featurizer.get_all_features(smiles)
   
   # Generate features for a dataset
   smiles_list = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]
   features_df, feature_names = featurizer.featurize_dataset(smiles_list)
   ```

3. **Configuration**
   All featurization parameters are stored in `config.py`:
   - `MORGAN_FP_RADIUS`: Radius for Morgan fingerprints
   - `MORGAN_FP_BITS`: Number of bits for fingerprint vectors
   - `ATOM_PAIR_FP_BITS`: Number of bits for atom pair fingerprints
   - `TORSION_FP_BITS`: Number of bits for torsion fingerprints
   - `RANDOM_SEED`: Random seed for reproducibility

4. **Reproducibility**
   The featurization process is fully reproducible:
   - Fixed random seeds for all operations
   - Consistent parameter values across runs
   - Version-controlled package dependencies
   - Centralized configuration

5. **Error Handling**
   The featurizer includes robust error handling:
   - Invalid SMILES validation
   - Graceful handling of failed feature generation
   - Logging of errors and warnings
   - Replacement of failed features with zeros

### Model Architecture
- **Algorithm**: XGBoost Classifier
- **Random State**: 42 (for reproducibility)
- **Train-Test Split**: 80-20 split with random state 42
- **Feature Scaling**: StandardScaler for normalization

### Performance Metrics
The model achieves the following performance metrics:
- Accuracy: 94.83%
- Precision: 95.05%
- Recall: 98.97%
- F1 Score: 96.97%
- ROC AUC: 98.05%

### Running the Model
To train the model and generate visualizations, run:
```bash
python scripts/run_workflow.py
```

This script will:
1. Create necessary directories (`data/` and `models/`)
2. Load and preprocess the dataset
3. Generate molecular features
4. Train the XGBoost model
5. Evaluate model performance
6. Generate visualizations

### Output Files
The following files are generated:
- `data/visualization_data.pkl`: Contains test data and predictions
- `models/hia_model.joblib`: Trained XGBoost model
- `models/feature_scaler.joblib`: Feature scaler for preprocessing
- `data/performance_metrics.png`: Bar chart of model metrics
- `data/confusion_matrix.png`: Confusion matrix visualization
- `data/roc_curve.png`: ROC curve with AUC score
- `data/feature_importance.png`: Top 20 most important features

### Visualization Interpretation
1. **Performance Metrics**: Shows the model's accuracy, precision, recall, F1 score, and ROC AUC
2. **Confusion Matrix**: Displays true positives, true negatives, false positives, and false negatives
3. **ROC Curve**: Illustrates the trade-off between true positive rate and false positive rate
4. **Feature Importance**: Identifies the most influential molecular features for absorption prediction

## Executing the Code

### Configuration
The project uses a centralized configuration file (`config.py`) that contains all important parameters:
- Dataset parameters (test size, random seed)
- Feature generation parameters (fingerprint sizes, radii)
- Model parameters (XGBoost hyperparameters)
- Visualization settings
- File paths

This ensures reproducibility and makes it easy to modify parameters without changing the code.

1. **Download the Dataset**
```bash
python scripts/download_dataset.py
```
This will:
- Download the HIA_Hou dataset from TDC
- Save it as `data/hia_hou_dataset.csv`
- Generate dataset statistics in `data/hia_hou_info.txt`

2. **Generate Features**
```bash
python scripts/featurize_molecules.py
```
This will:
- Generate all molecular features
- Save features to `data/molecular_features.csv`
- Log any errors or warnings to the console

3. **Train the Model**
```bash
python scripts/run_workflow.py
```
This will:
- Load and preprocess the dataset
- Generate molecular features
- Train the XGBoost model
- Evaluate performance
- Generate visualizations

### Output Files and Formats

1. **Dataset Files**:
   - `data/hia_hou_dataset.csv`: Original dataset (columns: Drug_ID, Drug, Y)
   - `data/hia_hou_info.txt`: Dataset statistics and information
   - `data/molecular_features.csv`: Generated features (CSV format)

2. **Model Files**:
   - `models/hia_model.joblib`: Trained XGBoost model
   - `models/feature_scaler.joblib`: Feature scaler for preprocessing
   - `data/model_performance.txt`: Model evaluation metrics

3. **Visualization Files**:
   - `data/performance_metrics.png`: Bar chart of model metrics
   - `data/confusion_matrix.png`: Confusion matrix visualization
   - `data/roc_curve.png`: ROC curve with AUC score
   - `data/feature_importance.png`: Top 20 most important features

### Error Handling and Logging
The project includes comprehensive error handling:
- Invalid SMILES strings are logged and handled gracefully
- Failed feature generation is logged with detailed error messages
- All operations are logged with appropriate severity levels
- Configuration errors are caught and reported clearly

## Contributing
1. Fork the repository
2. Make your feature a new branch.
3. Make the necessary adjustments.
4. Submit a pull request

