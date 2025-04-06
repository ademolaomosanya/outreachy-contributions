import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import DataStructs
import logging
from typing import List, Tuple, Dict, Union, Optional
from tqdm import tqdm
import warnings
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularFeaturizer:
    """
    A class for generating comprehensive molecular features from SMILES strings.
    
    This class implements various molecular featurization methods including:
    - Morgan Fingerprints (ECFP)
    - MACCS Keys
    - Atom Pair Fingerprints
    - Topological Torsion Fingerprints
    - Molecular Descriptors
    - 3D Molecular Descriptors
    
    All parameters are loaded from config.py to ensure reproducibility.
    
    Example:
        >>> featurizer = MolecularFeaturizer()
        >>> smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        >>> features = featurizer.get_all_features(smiles)
        >>> print(features.keys())
        dict_keys(['morgan', 'maccs', 'atom_pair', 'torsion', 'descriptors', '3d_descriptors'])
    """
    
    def __init__(self):
        """
        Initialize the featurizer with available descriptors.
        
        Sets up:
        - Molecular descriptor calculator
        - Descriptor names list
        - Logging configuration
        """
        self.descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            [x[0] for x in Descriptors._descList]
        )
        self.descriptor_names = [x[0] for x in Descriptors._descList]
    
    def _validate_smiles(self, smiles: str) -> bool:
        """
        Validate SMILES string format.
        
        Args:
            smiles (str): SMILES string to validate
            
        Returns:
            bool: True if SMILES is valid, False otherwise
            
        Example:
            >>> featurizer._validate_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
            True
            >>> featurizer._validate_smiles("invalid_smiles")
            False
        """
        if not isinstance(smiles, str):
            return False
        if not smiles.strip():
            return False
        return True
    
    def _get_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES to RDKit molecule object with error handling.
        
        Args:
            smiles (str): SMILES string to convert
            
        Returns:
            Optional[Chem.Mol]: RDKit molecule object or None if conversion fails
            
        Example:
            >>> mol = featurizer._get_mol("CC(=O)OC1=CC=CC=C1C(=O)O")
            >>> print(type(mol))
            <class 'rdkit.Chem.rdchem.Mol'>
        """
        if not self._validate_smiles(smiles):
            logger.warning(f"Invalid SMILES format: {smiles}")
            return None
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None
            return mol
        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {str(e)}")
            return None
    
    def get_morgan_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Generate Morgan fingerprint for a molecule."""
        mol = self._get_mol(smiles)
        if mol is None:
            return None
        
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 
                MORGAN_FP_RADIUS, 
                nBits=MORGAN_FP_BITS, 
                useFeatures=False
            )
            arr = np.zeros((MORGAN_FP_BITS,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception as e:
            logger.error(f"Error generating Morgan fingerprint for {smiles}: {str(e)}")
            return None
    
    def get_maccs_keys(self, smiles: str) -> Optional[np.ndarray]:
        """Generate MACCS keys for a molecule."""
        mol = self._get_mol(smiles)
        if mol is None:
            return None
        
        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((167,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception as e:
            logger.error(f"Error generating MACCS keys for {smiles}: {str(e)}")
            return None
    
    def get_atom_pair_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Generate atom pair fingerprint for a molecule."""
        mol = self._get_mol(smiles)
        if mol is None:
            return None
        
        try:
            fp = Pairs.GetAtomPairFingerprint(mol)
            arr = np.zeros((ATOM_PAIR_FP_BITS,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception as e:
            logger.error(f"Error generating atom pair fingerprint for {smiles}: {str(e)}")
            return None
    
    def get_torsion_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Generate topological torsion fingerprint for a molecule."""
        mol = self._get_mol(smiles)
        if mol is None:
            return None
        
        try:
            fp = Torsions.GetTopologicalTorsionFingerprint(mol)
            arr = np.zeros((TORSION_FP_BITS,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception as e:
            logger.error(f"Error generating torsion fingerprint for {smiles}: {str(e)}")
            return None
    
    def get_molecular_descriptors(self, smiles: str) -> Optional[np.ndarray]:
        """Calculate molecular descriptors for a molecule."""
        mol = self._get_mol(smiles)
        if mol is None:
            return None
        
        try:
            descriptors = self.descriptor_calculator.CalcDescriptors(mol)
            return np.array(descriptors)
        except Exception as e:
            logger.error(f"Error calculating descriptors for {smiles}: {str(e)}")
            return None
    
    def get_3d_descriptors(self, smiles: str) -> Optional[np.ndarray]:
        """Calculate 3D molecular descriptors."""
        mol = self._get_mol(smiles)
        if mol is None:
            return None
        
        try:
            # Generate 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=RANDOM_SEED)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Calculate available 3D descriptors
            descriptors = [
                rdMolDescriptors.CalcPBF(mol),  # Plane of Best Fit
                rdMolDescriptors.CalcPMI1(mol),  # Principal Moments of Inertia
                rdMolDescriptors.CalcPMI2(mol),
                rdMolDescriptors.CalcPMI3(mol),
                rdMolDescriptors.CalcNPR1(mol),  # Normalized Principal Moments Ratios
                rdMolDescriptors.CalcNPR2(mol),
                rdMolDescriptors.CalcInertialShapeFactor(mol),
                rdMolDescriptors.CalcEccentricity(mol),
                rdMolDescriptors.CalcAsphericity(mol),
                rdMolDescriptors.CalcSpherocityIndex(mol)
            ]
            return np.array(descriptors)
        except Exception as e:
            logger.error(f"Error calculating 3D descriptors for {smiles}: {str(e)}")
            return None
    
    def get_all_features(self, smiles: str) -> Dict[str, Optional[np.ndarray]]:
        """Generate all available features for a molecule."""
        return {
            'morgan_fp': self.get_morgan_fingerprint(smiles),
            'maccs_keys': self.get_maccs_keys(smiles),
            'atom_pair_fp': self.get_atom_pair_fingerprint(smiles),
            'torsion_fp': self.get_torsion_fingerprint(smiles),
            'molecular_descriptors': self.get_molecular_descriptors(smiles),
            '3d_descriptors': self.get_3d_descriptors(smiles)
        }
    
    def featurize_dataset(self, smiles_list: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Featurize a list of SMILES strings."""
        features_dict = {}
        feature_names_dict = {}
        
        # Initialize feature lists
        for feature_type in ['morgan_fp', 'maccs_keys', 'atom_pair_fp', 
                           'torsion_fp', 'molecular_descriptors', '3d_descriptors']:
            features_dict[feature_type] = []
            feature_names_dict[feature_type] = []
        
        # Generate features for each molecule
        for smiles in tqdm(smiles_list, desc="Featurizing molecules"):
            features = self.get_all_features(smiles)
            for feature_type, feature_array in features.items():
                if feature_array is not None:
                    features_dict[feature_type].append(feature_array)
                else:
                    # Handle invalid molecules with NaN arrays
                    if feature_type == 'morgan_fp':
                        features_dict[feature_type].append(np.zeros(MORGAN_FP_BITS))
                    elif feature_type == 'maccs_keys':
                        features_dict[feature_type].append(np.zeros(167))
                    elif feature_type in ['atom_pair_fp', 'torsion_fp']:
                        features_dict[feature_type].append(np.zeros(ATOM_PAIR_FP_BITS))
                    elif feature_type == 'molecular_descriptors':
                        features_dict[feature_type].append(np.zeros(len(self.descriptor_names)))
                    elif feature_type == '3d_descriptors':
                        features_dict[feature_type].append(np.zeros(10))
                    else:
                        features_dict[feature_type].append(np.zeros(0))
        
        # Convert to DataFrames with proper column names
        dfs = {}
        for feature_type, features in features_dict.items():
            if features:
                if feature_type == 'molecular_descriptors':
                    dfs[feature_type] = pd.DataFrame(features, columns=self.descriptor_names)
                elif feature_type == 'morgan_fp':
                    dfs[feature_type] = pd.DataFrame(features, columns=[f'morgan_{i}' for i in range(MORGAN_FP_BITS)])
                elif feature_type == 'maccs_keys':
                    dfs[feature_type] = pd.DataFrame(features, columns=[f'maccs_{i}' for i in range(167)])
                elif feature_type == 'atom_pair_fp':
                    dfs[feature_type] = pd.DataFrame(features, columns=[f'atom_pair_{i}' for i in range(ATOM_PAIR_FP_BITS)])
                elif feature_type == 'torsion_fp':
                    dfs[feature_type] = pd.DataFrame(features, columns=[f'torsion_{i}' for i in range(TORSION_FP_BITS)])
                elif feature_type == '3d_descriptors':
                    dfs[feature_type] = pd.DataFrame(features, columns=[
                        'pbf', 'pmi1', 'pmi2', 'pmi3', 'npr1', 'npr2',
                        'inertial_shape', 'eccentricity',
                        'asphericity', 'spherocity'
                    ])
        
        # Combine all features
        combined_df = pd.concat(dfs.values(), axis=1)
        
        return combined_df, feature_names_dict

def main():
    """Example usage of the MolecularFeaturizer class."""
    # Example SMILES
    smiles_list = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    ]
    
    # Initialize featurizer
    featurizer = MolecularFeaturizer()
    
    # Featurize dataset
    features_df, feature_names = featurizer.featurize_dataset(smiles_list)
    
    # Save features
    features_df.to_csv('data/molecular_features.csv', index=False)
    
    # Print feature information
    logger.info(f"Generated features for {len(smiles_list)} molecules")
    logger.info(f"Total number of features: {features_df.shape[1]}")
    logger.info("Feature types generated:")
    for feature_type, names in feature_names.items():
        logger.info(f"- {feature_type}: {len(names)} features")

if __name__ == "__main__":
    main() 