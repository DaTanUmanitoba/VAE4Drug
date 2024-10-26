import json
import os
import argparse
import csv
import random
from model.utils import combine_fragments
import rdkit.Chem as Chem

# Disable RDKit warnings
from rdkit import RDLogger
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)

def read_data(data_path):
    with open(data_path, 'r') as data_file:
        return json.load(data_file)

def read_csv(data_path):
    rows = []
    with open(data_path, newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            rows.append(row)
    return rows

def sanitize_molecule(mol):
    """
    Sanitize molecule SMILES and return sanitized SMILES or None if sanitization fails.
    """
    try:
        Chem.SanitizeMol(Chem.MolFromSmiles(mol))
        return mol
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Merge molecular fragments with backbones to generate new molecules.")
    parser.add_argument('--filtered_frag_path', type=str, required=True, help="Path to the input backbone CSV file.")
    parser.add_argument('--vqvae_frags', type=str, required=True, help="Path to the input VQ-VAE fragment JSON file.")
    parser.add_argument('--output_dir', type=str, required=True, default='./', help="Output directory for the results.")
    parser.add_argument('--n_samples', type=int, required=True, default=10, help="Number of samples per backbone to generate.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    out_file = os.path.join(args.output_dir, 'merged_molecules.csv')
    print(f'Output file: {out_file}')

    # Read and process data
    all_piecies = read_csv(args.filtered_frag_path)
    print(all_piecies[0])
    frag_dict = {piece[1]: piece[0] for piece in all_piecies}# if piece[3] == '1'}

    print(len(frag_dict), "active molecules found.")

    frags = read_data(args.vqvae_frags)
    print(frags[1:3])

    all_new_molecules = []
    for i, bb in enumerate(frag_dict):
        if i % 1000 == 0:
            print(f'Processing {i}th backbone out of {len(frag_dict)}')

        random.seed(42 + i)  # Set seed for reproducibility
        rand_inds = random.sample(range(len(frags)), args.n_samples)
        rand_frags = [frags[x] for x in rand_inds]

        valid_frags = [sanitize_molecule(frag) for frag in rand_frags if frag != '*' and sanitize_molecule(frag)]

        for frag in valid_frags:
            new_mol = combine_fragments(bb, frag)
            if new_mol:
                all_new_molecules.append([new_mol, bb, frag, frag_dict[bb]])

    print(f'Total new molecules generated: {len(all_new_molecules)}')

    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['new_mol', 'backbone', 'vqvae_frag', 'orig_mol'])
        writer.writerows(all_new_molecules)

if __name__ == '__main__':
    main()