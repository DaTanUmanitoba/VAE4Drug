import numpy as np
import pandas as pd
import os
import csv
import pickle
import argparse
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

def load_csv_data(filepath):
    """Load CSV data from file and return rows excluding header."""
    records = []
    with open(filepath, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip header
        for row in csvreader:
            records.append(row)
    return records

def compute_fingerprint(smiles, radius=2, n_bits=2048):
    """Compute RDKit fingerprint from a SMILES string."""
    molecule = Chem.MolFromSmiles(smiles)
    fingerprint_obj = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=radius, nBits=n_bits)
    fingerprint_arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint_obj, fingerprint_arr)
    return fingerprint_arr

def train_classifier(positive_samples, negative_samples, test_split=0.2, estimators=100, depth=2):
    """Train a RandomForest classifier and evaluate its performance."""
    # Generate fingerprints for positive and negative samples
    pos_fps = [compute_fingerprint(smile) for smile in positive_samples]
    neg_fps = [compute_fingerprint(smile) for smile in negative_samples]
    
    # Split data into training and testing sets
    total_samples = len(pos_fps) + len(neg_fps)
    test_size = int(test_split * total_samples)
    pos_test_end = int(test_size * (len(pos_fps) / total_samples))
    neg_test_end = int(test_size * (len(neg_fps) / total_samples))
    
    train_fps = pos_fps[pos_test_end:] + neg_fps[neg_test_end:]
    test_fps = pos_fps[:pos_test_end] + neg_fps[:neg_test_end]
    train_labels = np.concatenate([np.ones(len(pos_fps[pos_test_end:])), np.zeros(len(neg_fps[neg_test_end:]))])
    test_labels = np.concatenate([np.ones(pos_test_end), np.zeros(neg_test_end)])
    
    # Weight samples to address imbalance
    weights = [1 if label == 0 else len(neg_fps) / len(pos_fps) for label in train_labels]
    
    # Train RandomForest classifier
    model = RandomForestClassifier(n_estimators=estimators, max_depth=depth, random_state=0)
    model.fit(train_fps, train_labels, sample_weight=weights)
    
    # Evaluate model
    predicted_probs = model.predict_proba(test_fps)
    auc_score = roc_auc_score(test_labels, predicted_probs[:, 1])
    fpr, tpr, _ = roc_curve(test_labels, predicted_probs[:, 1])
    
    return model, auc_score, fpr, tpr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path of the training data')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory for the classifier')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data_path = args.output_dir + '/rf_model.pkl'
    data_frame = pd.read_csv(args.data)
    data_frame.columns = ['smiles', 'activity']
    
    positive_smiles = data_frame[data_frame['activity'] == 1]['smiles'].tolist()
    negative_smiles = data_frame[data_frame['activity'] == 0]['smiles'].tolist()
    
    classifier, roc_score, _, _ = train_classifier(positive_smiles, negative_smiles)
    print(f'Model AUC Score: {roc_score}')
    
    with open(data_path, 'wb') as model_file:
        pickle.dump(classifier, model_file)

if __name__ == '__main__':
    main()
