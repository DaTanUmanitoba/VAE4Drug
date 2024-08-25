import pickle
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def compute_fingerprint(smiles: str):
    """ Generate the Morgan fingerprint for the given SMILES representation. """
    molecule = Chem.MolFromSmiles(smiles)
    fingerprint_obj = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048, useChirality=False)
    fingerprint_vector = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint_obj, fingerprint_vector)
    return fingerprint_vector

class GSK3Classifier():
    """ Predicts activity based on classifier for GSK3 inhibitors. """
    
    def __init__(self, classifier_path ):
        with open(classifier_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, smiles_list):
        """ Evaluate a list of SMILES and return predictions for their activities. """
        features = []
        validity_mask = []
        for smiles in smiles_list:
            molecule = Chem.MolFromSmiles(smiles)
            if not molecule:
                print(f'Invalid SMILES: {smiles}')
            fingerprint = compute_fingerprint(smiles) if molecule else np.zeros((1,2048))
            validity_mask.append(int(molecule is not None))
            features.append(fingerprint)
        
        predictions = self.model.predict_proba(features)[:, 1]
        predictions *= np.array(validity_mask)
        return predictions

class JNK3Classifier():
    """ Predicts activity based on classifier for JNK3 inhibitors. """
    
    def __init__(self, classifier_path):
        with open(classifier_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, smiles_list):
        """ Evaluate a list of SMILES and return predictions. """
        features = []
        validity_mask = []
        for smiles in smiles_list:
            molecule = Chem.MolFromSmiles(smiles)
            fingerprint = compute_fingerprint(smiles) if molecule else np.zeros((1,2048))
            validity_mask.append(int(molecule is not None))
            features.append(fingerprint)

        predictions = self.model.predict_proba(features)[:, 1]
        predictions *= np.array(validity_mask)
        return predictions
