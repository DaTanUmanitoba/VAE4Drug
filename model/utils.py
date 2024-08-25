import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem

def combine_tensors(tensors):
    """
    Aligns a list of tensors by padding and then combines them along a new dimension.
    """
    if not tensors:
        raise ValueError("Input tensor list is empty.")
    
    # Find the maximum length in the tensors for uniform size
    max_length = max(tensor.size(0) for tensor in tensors)
    adjusted_tensors = [F.pad(tensor, (0, 0, 0, max_length - tensor.size(0))) for tensor in tensors]
    return torch.stack(adjusted_tensors, dim=0)

def select_indices(source_tensor, axis, indices):
    """
    Selects elements from a multi-dimensional tensor using specified indices along a given dimension.
    """
    indices_shape = indices.size()
    additional_dims = source_tensor.size()[1:]
    final_shape = indices_shape + additional_dims
    selected = source_tensor.index_select(axis, indices.view(-1))
    return selected.view(final_shape)

def generate_fingerprint(smiles_string, radius=2, dimensions=2048, chirality=False):
    """
    Generates a molecular fingerprint from a SMILES string.
    """
    molecule = Chem.MolFromSmiles(smiles_string)
    fingerprint = None
    if molecule is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=radius, nBits=dimensions, useChirality=chirality)
    return fingerprint

def discard_dummy_atoms(molecule):
    """
    Removes dummy atoms ('*') from a molecule and returns the modified molecule and the index of the removed atom.
    """
    dummy_index = 0
    has_dummy = False
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == '*':
            dummy_index = atom.GetIdx()
            has_dummy = True
            break
    modifiable_mol = Chem.RWMol(molecule)
    if has_dummy:
        modifiable_mol.RemoveAtom(dummy_index)
    return modifiable_mol.GetMol(), dummy_index

def combine_fragments(smiles1, smiles2):
    """
    Combines two molecular fragments represented by SMILES, removing dummy atoms and creating a bond.
    """
    fragment_one = Chem.MolFromSmiles(smiles1)
    fragment_two = Chem.MolFromSmiles(smiles2)
    clean_fragment_one, index_one = discard_dummy_atoms(fragment_one)
    clean_fragment_two, index_two = discard_dummy_atoms(fragment_two)
    
    combined = Chem.CombineMols(clean_fragment_one, clean_fragment_two)
    editable_combined = Chem.EditableMol(combined)
    editable_combined.AddBond(index_one, clean_fragment_one.GetNumAtoms() + index_two, order=Chem.BondType.SINGLE)
    
    final_combined = editable_combined.GetMol()
    try:
        Chem.SanitizeMol(final_combined)
        return Chem.MolToSmiles(final_combined)
    except Exception as e:
        #print(f'Sanitization problem: {e}')
        return None
