import selfies as sf
from rdkit import Chem
from torch.utils.data import Dataset
from model.mol_graph.mol_graph import MoleculeGraph

class FragmentData(Dataset):
    """
    Dataset preparation for processing chemical fragment data into SELFIES and graph tensors for training.
    """
    def __init__(self, side_chain, labels, atom_vocab, selfies_indx, batch_size):
        """
        Initializes the dataset object.

        Parameters:
        - side_chain: List of fragment structures as SMILES.
        - labels: List of labels for the data.
        - atom_vocab: vocabulary for atoms.
        - selfies_indx: SELFIES string index mapped from the SELFILE string.
        - batch_size: Number of samples in each batch.
        """
        self.batch_fragments = [side_chain[i:i+batch_size] for i in range(0, len(side_chain), batch_size)]
        self.batch_labels = [labels[i:i+batch_size] for i in range(0, len(labels), batch_size)]
        self.atom_vocab = atom_vocab
        self.selfies_ind = selfies_indx

    def __len__(self):
        return len(self.batch_fragments)

    def __getitem__(self, idx):
        """
        Returns the encoded graphs, selfies, and labels for a batch.
        """
        return self.encode_fragments(self.batch_fragments[idx], self.batch_labels[idx])

    def encode_fragments(self, frag_smiles, labels):
        """
        Encodes fragments into tuple of tensors and other data structures for model input
        Parameters:
        - frag_smiles: list of fragment SMILES for the batch.
        - labels: labels for the batch.
        """
        valid_indices = [i for i, smile in enumerate(frag_smiles) if Chem.MolFromSmiles(smile)]
        valid_smiles = [frag_smiles[i] for i in valid_indices]
        valid_selfies = [sf.encoder(smile) for smile in valid_smiles]
        valid_labels = [labels[i] for i in valid_indices]

        pad_len = max(sf.len_selfies(s) for s in valid_selfies) + 1
        selfies_enc = [sf.selfies_to_encoding(s, vocab_stoi=self.selfies_ind, pad_to_len=pad_len, enc_type='label') for s in valid_selfies]
        selfies_mask = [[0 if x == self.selfies_ind['[nop]'] else 1 for x in enc] for enc in selfies_enc]
        for mask in selfies_mask:
            mask[0] = 1 

        graph_tensors = MoleculeGraph.tensorize(valid_smiles, self.atom_vocab, only_graph=True)

        return graph_tensors, valid_selfies, selfies_enc, selfies_mask, valid_labels
