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

