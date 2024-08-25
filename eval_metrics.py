import argparse
import numpy as np
import pandas as pd
import os
import csv
import json
from rdkit import Chem, DataStructs
import rdkit.Chem.QED as QED
from p_tqdm import p_umap
from model.rf_scoring import GSK3Classifier, JNK3Classifier
from model.utils import generate_fingerprint

def clean_smiles(smiles):
    """Attempt to clean and validate SMILES strings using RDKit."""
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        return None

def check_smiles_valid(df):
    """Check and mark validity of SMILES in a DataFrame."""
    for index, row in df.iterrows():
        if clean_smiles(row['new_mol']) is None:
            print("Found invalid new mol!")
            df.at[index, 'generated_valid'] = False
        if clean_smiles(row['orig_mol']) is None:
            print("Found invalid original mol!")
            df.at[index, 'orig_valid'] = False
    return df

def read_data(data_path):
    """Read data from CSV file into a list of rows."""
    rows = []
    with open(data_path, newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            rows.append(row)
    return rows

def pair_wise_tanimoto(smiles1, smiles2):
    """Calculate Tanimoto similarity between two SMILES strings."""
    fp1 = generate_fingerprint(smiles1)
    fp2 = generate_fingerprint(smiles2)
    return DataStructs.FingerprintSimilarity(fp1, fp2)

def calculate_distance(df):
    """Calculate property improvement and Tanimoto similarity."""
    for index, row in df.iterrows():
        df.at[index, 'property_improv'] = row['Class_new'] - row['Class']
        if row['generated_valid'] and row['orig_valid']:
            df.at[index, 'tanimoto_sim'] = pair_wise_tanimoto(row['orig_mol'], row['new_mol'])
    return df

def get_novelty(ori_smiles, generated_smiles, threshold):
    """Calculate novelty based on similarity threshold."""
    sim = 0
    ori_fps = [generate_fingerprint(x) for x in ori_smiles]
    generated_fps = [generate_fingerprint(x.lstrip('*')) for x in generated_smiles]

    for fps in generated_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fps, ori_fps)
        if max(sims) >= threshold:
            sim += 1
    novelty = 1 - sim / len(generated_fps)
    return novelty, generated_fps

def get_diversity(fps):
    """Calculate diversity of fingerprints using parallel computation."""
    length = len(fps)
    inputs = [(fps[i], fps[:i]) for i in range(length)]

    def bulk_sim(fps_inputs):
        fps1, fps_rest = fps_inputs
        return DataStructs.BulkTanimotoSimilarity(fps1, fps_rest)

    outputs = p_umap(bulk_sim, inputs)
    outputs = sum(outputs, [])
    sum_scores = sum(outputs)
    n_pairs = length * (length - 1) / 2
    diversity = 1 - sum_scores / n_pairs
    return diversity

def get_qed(smiles_list):
    """Compute the QED scores for a list of SMILES strings."""
    scores = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        scores.append(QED.qed(mol) if mol else 0)
    return scores

def main():
    parser = argparse.ArgumentParser(description='evaluation metrics for the generated molecules.')
    parser.add_argument('--origin_data', type=str, default="", help='prediction data of the orinial molecule, only used for BACE dataset')
    parser.add_argument('--generated_data', type=str, required=True, help='the generated molecules')
    parser.add_argument('--output_path', type=str, required=True, help='directory of the output')
    parser.add_argument('--novelty_threshold', type=float, default=0.6, help='threshold value above which the novelty score is compared')
    parser.add_argument('--dataset', type=str, required=True, default='bace', help='distinguish BACE/JNK3/GSK3, used to load the corresponding oracle predictor')
    parser.add_argument('--classifier_dir', type=str, required=True, help='the directory of the predictor')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Load and process datasets
    if args.dataset in ['gsk3', 'jnk3']:
        df = pd.read_csv(args.generated_data)
        model = GSK3Classifier(args.classifier_dir) if args.dataset == 'gsk3' else JNK3Classifier(args.classifier_dir)
        df['Class_new'] = model.predict(df['new_mol'].tolist())
        df['Class'] = model.predict(df['orig_mol'].tolist())
    else:
        ori_data = pd.read_csv(args.origin_data)
        generated_data = pd.read_csv(args.generated_data)
        ori_data.columns = ['orig_mol', 'label', 'Class']
        generated_data.columns = ['new_mol', 'backbone', 'vqvae_frag', 'orig_mol', 'Class_new']
        df = pd.merge(generated_data, ori_data, on='orig_mol', how='inner')

    df = df.loc[df.groupby('orig_mol')['Class_new'].idxmax()]
    print(f'dimension of the data and its columns: {df.shape}  {df.columns}')
    
    # Further data processing
    df['orig_valid'] = True
    df['generated_valid'] = True
    df['property_improv'] = 0
    df['tanimoto_sim'] = 0
    
    df = check_smiles_valid(df)
    df = calculate_distance(df)

    
    novelty, generated_fps = get_novelty(df['orig_mol'], df['new_mol'], args.novelty_threshold)
    diversity = get_diversity(generated_fps)
    qed = get_qed(df['new_mol'])
    df['ori_qed'] = get_qed(df['orig_mol'])
    df['new_qed'] = qed
    
    df_out = args.output_path + '/df.csv'
    df.to_csv(df_out, index=False)
    
    # Output results
    stats = {
        'Average_Improvement': df['property_improv'].mean(),
        'Average_similarity': df['tanimoto_sim'].mean(),
        'Average_QED': np.mean(qed),
        'Novelty': novelty,
        'Diversity': diversity
    }
    
    print(stats)
    with open(os.path.join(args.output_path, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == '__main__':
    main()
