import rdkit.Chem as Chem
import torch

class ChemicalVocabulary(object):
    """Represents a vocabulary of SMILES chemical representations."""
    
    def __init__(self, smiles_list):
        self.chemicals = list(smiles_list)  # Initialize with list of SMILES
        self.index_map = {smile: index for index, smile in enumerate(self.chemicals)}

    def __getitem__(self, smile):
        """Retrieve the index of a SMILE string from the vocabulary."""
        return self.index_map[smile]

    def get_smile(self, index):
        """Return the SMILES at the specified index."""
        return self.chemicals[index]

    def vocabulary_size(self):
        """Return the size of the vocabulary."""
        return len(self.chemicals)

class PairedChemicalVocabulary(object):
    """Handles a vocabulary of paired SMILES for molecular interactions."""
    
    def __init__(self, smile_pairs, use_cuda=True):
        head_smiles = next(zip(*smile_pairs))
        self.head_vocab = list(set(head_smiles))
        self.head_index_map = {smile: idx for idx, smile in enumerate(self.head_vocab)}

        self.pairs = [tuple(pair) for pair in smile_pairs]
        self.interaction_sizes = [count_interactions(pair[1]) for pair in self.pairs]
        self.pair_index_map = {pair: idx for idx, pair in enumerate(self.pairs)}

        self.interaction_mask = torch.zeros(len(self.head_vocab), len(self.pairs))
        for head, tail in smile_pairs:
            head_id = self.head_index_map[head]
            pair_id = self.pair_index_map[(head, tail)]
            self.interaction_mask[head_id, pair_id] = 1000.0

        if use_cuda: 
            self.interaction_mask = self.interaction_mask.cuda()
        self.interaction_mask -= 1000.0

    def __getitem__(self, pair):
        """Retrieve the indices of a head and its corresponding pair."""
        assert isinstance(pair, tuple), "Input must be a tuple"
        return self.head_index_map[pair[0]], self.pair_index_map[pair]

    def get_head_smile(self, index):
        """Return the head SMILE at the specified index."""
        return self.head_vocab[index]

    def get_tail_smile(self, index):
        """Return the tail SMILE at the specified index."""
        return self.pairs[index][1]

    def vocabulary_size(self):
        """Return the size of the head and pair vocabularies."""
        return len(self.head_vocab), len(self.pairs)

    def retrieve_interaction_mask(self, head_index):
        """Select the interaction mask for a given head index."""
        return self.interaction_mask.index_select(index=head_index, dim=0)

    def get_interaction_size(self, pair_index):
        """Retrieve the interaction size for the specified pair index."""
        return self.interaction_sizes[pair_index]

def count_interactions(smile_string):
    """Counts the number of interacting atoms in a SMILE string."""
    molecule = Chem.MolFromSmiles(smile_string)
    interactions = [atom for atom in molecule.GetAtoms() if atom.GetAtomMapNum() > 0]
    return max(1, len(interactions))

ATOMS_WITH_VALENCE = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1), ('C', -1), ('Cl', 0), ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0), ('F', 1), ('F', -1), ('I', -1), ('I', 0), ('I', 1), ('I', 2), ('I', 3), ('N', 0), ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1), ('P', 0), ('P', 1), ('P', -1), ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0), ('Si', -1), ('*', 0)]
atom_vocabulary = ChemicalVocabulary(ATOMS_WITH_VALENCE)

VALENCE_LIMITS = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':3, 'O':2, 'P':3, 'S':4, 'Se':4, 'Si':4}

