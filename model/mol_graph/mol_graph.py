import torch
import rdkit.Chem as Chem
import networkx as nx
from collections import deque
from .vocab import VALENCE_LIMITS

# Simple addition function for either integers or tuples
sum_elements = lambda x, y: x + y if isinstance(x, int) else (x[0] + y, x[1] + y)

class MoleculeGraph(object):
    """Create a graph embedding for a molecule, define the initial encodings for atoms and bones"""
    
    BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    MAX_POSITION = 40

    def __init__(self, chemical_smile):
        self.chemical_smile = chemical_smile
        self.molecule = Chem.MolFromSmiles(chemical_smile)
        self.graph = self.create_graph()
        self.bfs_order = self.bfs()

    def create_graph(self):
        """Constructs molecular graph from a molecule"""
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.molecule))
        for atom in self.molecule.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())

        for bond in self.molecule.GetBonds():
            source = bond.GetBeginAtom().GetIdx()
            target = bond.GetEndAtom().GetIdx()
            bond_type = MoleculeGraph.BOND_TYPES.index(bond.GetBondType())
            graph[source][target]['label'] = bond_type
            graph[target][source]['label'] = bond_type

        return graph

    def bfs(self):
        """Breadth-first search to order nodes for processing"""
        order = []
        visited = {0}
        self.graph.nodes[0]['position'] = 0
        root_atom = self.molecule.GetAtomWithIdx(0)
        queue = deque([root_atom])

        while queue:
            current = queue.popleft()
            current_idx = current.GetIdx()
            for neighbor in current.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in visited:
                    continue

                current_front = [current_idx] + [atom.GetIdx() for atom in queue]
                bond_types = [0] * len(current_front)
                neighbor_ids = {nei.GetIdx() for nei in neighbor.GetNeighbors()}

                for i, atom_idx in enumerate(current_front):
                    if atom_idx in neighbor_ids:
                        bond_types[i] = self.graph[neighbor_idx][atom_idx]['label']

                order.append((current_idx, neighbor_idx, current_front, bond_types))
                self.graph.nodes[neighbor_idx]['position'] = min(MoleculeGraph.MAX_POSITION - 1, len(visited))
                visited.add(neighbor_idx)
                queue.append(neighbor)

            order.append((current_idx, None, None, None))

        return order

    @staticmethod
    def tensorize_actions(batch_of_molecules, vocabulary):
        batch_of_molecules = [MoleculeGraph(smile) for smile in batch_of_molecules]
        graph_tensors, masks, _ = MoleculeGraph.convert_to_tensors(
            batch_of_molecules, vocabulary, compute_masks=True)
        return graph_tensors, masks

    @staticmethod
    def tensorize(batch_of_molecules, vocabulary, only_graph=False):
        batch_of_molecules = [MoleculeGraph(smile) for smile in batch_of_molecules]
        graph_tensors, batch_graphs = MoleculeGraph.convert_to_tensors(
            batch_of_molecules, vocabulary, compute_masks=False)

        if only_graph:
            return graph_tensors

        graph_scopes = graph_tensors[-1]
        combine_elements = lambda element, shift: None if element is None else element + shift
        combine_lists = lambda lists, shift: None if lists is None else [item + shift for item in lists]

        all_orders = []
        for index, molecule in enumerate(batch_of_molecules):
            shift = graph_scopes[index][0]
            molecule_order = [(pos + shift, combine_elements(next_pos, shift), combine_lists(frontier, shift), bond_types) for pos, next_pos, frontier, bond_types in molecule.bfs_order]
            all_orders.append(molecule_order)

        return batch_graphs, graph_tensors, all_orders

    @staticmethod
    def convert_to_tensors(batch_of_molecules, vocab, compute_masks=False):
        """
        Conversion of molecular graphs to tensor representation
        """
        
        batch_graphs = [mol.graph for mol in batch_of_molecules]
        node_features, messages = [None], [(0, 0, 0)]
        adjacency_list, bond_graph = [[]], [[]]
        scopes, edge_scopes = [], []
        edge_index = {}
        all_graphs = []

        for idx, G in enumerate(batch_graphs):
            node_offset = len(node_features)
            edge_offset = len(messages)

            scopes.append((node_offset, len(G)))
            edge_scopes.append((edge_offset, len(G.edges)))
            G = nx.convert_node_labels_to_integers(G, first_label=node_offset)
            all_graphs.append(G)
            node_features.extend([None for v in G.nodes])

            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = idx
                node_features[v] = (vocab[attr], G.nodes[v]['position'])
                adjacency_list.append([])

            for u, v, attr in G.edges(data='label'):
                messages.append((u, v, attr))
                edge_index[(u, v)] = eid = len(edge_index) + 1
                G[u][v]['message_idx'] = eid
                adjacency_list[v].append(eid)
                bond_graph.append([])

            for u, v in G.edges:
                eid = edge_index[(u, v)]
                for w in G.predecessors(u):
                    if w == v: continue
                    bond_graph[eid].append(edge_index[(w, u)])

        node_features[0] = node_features[1]
        node_features = torch.LongTensor(node_features)
        messages = torch.LongTensor(messages)
        adjacency_list = pad_tensor(adjacency_list)
        bond_graph = pad_tensor(bond_graph)

        if compute_masks:
            attach_masks = [torch.tensor(create_attachment_mask(graph.molecule)) for graph in batch_of_molecules]
            attach_masks = torch.nn.utils.rnn.pad_sequence(attach_masks, padding_value=0, batch_first=True).bool()

            delete_masks = (messages[:, 2] != MoleculeGraph.BOND_TYPES.index(Chem.rdchem.BondType.AROMATIC)).long()
            delete_masks = [delete_masks[start:start+length] for start, length in edge_scopes]
            delete_masks = torch.nn.utils.rnn.pad_sequence(delete_masks, padding_value=0, batch_first=True).bool()
            return (node_features, messages, adjacency_list, bond_graph, scopes), (attach_masks, delete_masks, edge_scopes), nx.union_all(all_graphs)
        else:
            return (node_features, messages, adjacency_list, bond_graph, scopes), nx.union_all(all_graphs)

def pad_tensor(lists):
    """
    Pads a list of integer lists to the length of the longest list, then converts to a PyTorch IntTensor.
    """
    # Determine the maximum length of the lists in the provided list
    max_length = max(len(single_list) for single_list in lists) + 1

    # Extend each list in the list to the maximum length
    for single_list in lists:
        padding_length = max_length - len(single_list)
        single_list.extend([0] * padding_length)  # Append zeros to reach the maximum length

    return torch.IntTensor(lists)  # Convert the padded list to a tensor of integers


def create_attachment_mask(molecule):
    # Compute possible attachment points on a molecule
    attachment_mask = []
    for atom in molecule.GetAtoms():
        atom_symbol = atom.GetSymbol()
        implicit_valence = atom.GetImplicitValence()
        maximum_valence = VALENCE_LIMITS[atom_symbol] if atom_symbol in VALENCE_LIMITS else 0
        if maximum_valence > 0 and implicit_valence > 0:
            attachment_mask.append(1)
        else:
            attachment_mask.append(0)
    return attachment_mask

