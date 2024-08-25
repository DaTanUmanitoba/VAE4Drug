import random
from rdkit import Chem
import argparse
import csv
import os
from rdkit import RDLogger

# Disable RDKit warnings
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)

def split_molecule_randomly(smiles_data, min_weight=10, max_weight=20, attempts=10, base_seed=42):
    """ Randomly splits a molecule based on SMILES, avoiding duplicate fragments. """
    molecule = Chem.MolFromSmiles(smiles_data[0])
    label = smiles_data[1]
    if molecule is None:
        return None

    outcome, seen_fragments = [], []
    for attempt in range(attempts):
        bond_indices = list(range(molecule.GetNumBonds()))
        if not bond_indices:
            return None

        random.seed(base_seed + attempt)
        selected_bond = random.choice(bond_indices)
        broken_molecule = Chem.FragmentOnBonds(molecule, [selected_bond], addDummies=False)
        broken_smiles = Chem.MolToSmiles(broken_molecule)
        pieces = broken_smiles.split('.')

        duplicates = False
        for piece in pieces:
            if piece not in seen_fragments:
                seen_fragments.append(piece)
            else:
                duplicates = True

        if duplicates:
            continue

        pieces_molecules = [Chem.MolFromSmiles(piece) for piece in pieces if piece]
        piece_weights = [piece_mol.GetNumAtoms() for piece_mol in pieces_molecules if piece_mol  is not None]

        if len(piece_weights) < 2 or any(w < min_weight for w in piece_weights) or all(w > max_weight for w in piece_weights):
            continue

        sorted_pieces = sorted([(w, s) for w, s in zip(piece_weights, pieces)], key=lambda x: x[0])
        outcome.append([smiles_data[0], sorted_pieces[1][1], sorted_pieces[0][1], label])

    return outcome if outcome else None

def main():
    """ Processes SMILES data, randomly split molecules and save the results. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input SMILES data')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.data_path, newline='') as data_file:
        data_reader = csv.reader(data_file)
        next(data_reader)  # Skip header
        smiles_data = [row for row in data_reader]

    all_splits, skipped = [], 0
    for index, data in enumerate(smiles_data):
        fragments = split_molecule_randomly(data, min_weight=5, max_weight=20, attempts=10, base_seed=(index + 42))
        if fragments:
            all_splits.extend(fragments)
        else:
            skipped += 1

    #save output to a fragment.csv file in the output folder
    output_path = os.path.join(args.output_dir, 'fragments.csv')
    with open(output_path, 'w', newline='') as out_file:
        csv_writer = csv.writer(out_file)
        csv_writer.writerow(['ori_mol', 'backbone', 'side_chain', 'label'])
        csv_writer.writerows(all_splits)

    print(f'Total: {len(smiles_data)}, Skipped: {skipped}, Valid Splits: {len(all_splits)}')

if __name__ == "__main__":
    main()
