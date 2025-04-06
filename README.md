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
pip install tdc pandas
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
The model uses two types of molecular features:
1. **Morgan Fingerprints**: 2048-bit binary vectors representing molecular substructures
2. **Molecular Descriptors**: Various physicochemical properties calculated using RDKit

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

### 1. Get the dataset
```bash Python scripts/download_dataset.py ```will download and prepare the dataset.
This will:
The HIA_Hou dataset can be downloaded from TDC, saved as a CSV file in the `data` folder, and its statistics can be generated in `data/hia_hou_info.txt`.


### Output Documents
Following script execution, the main dataset file is located at `data/hia_hou_dataset.csv`.
The file "data/hia_hou_info.txt`": Comprehensive facts and statistics from the dataset

## Contributing
1. Fork the repository
2. Make your feature a new branch.
3. Make the necessary adjustments.
4. Submit a pull request

