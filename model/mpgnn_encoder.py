import torch
import torch.nn as nn
from model.mol_graph.mol_graph import MoleculeGraph
from model.rnn import CustomLSTM
from model.utils import combine_tensors, select_indices

class MessagePassingEncoder(nn.Module):
    """Encodes molecular graphs using a message passing neural network architecture."""
    def __init__(self, input_dim, feature_dim, output_dim, num_layers):
        super(MessagePassingEncoder, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.transform = nn.Sequential(
            nn.Linear(feature_dim + output_dim, output_dim),
            nn.ReLU(),
        )
        self.lstm = CustomLSTM(input_dim, output_dim, num_layers)

    def forward(self, node_features, edge_features, adj_list, bond_list, mask=None):
        edge_hidden = self.lstm(edge_features, bond_list)
        edge_hidden = self.lstm.state_projection(edge_hidden)
        neighbor_message = select_indices(edge_hidden, 0, adj_list).sum(dim=1)
        concatenated_features = torch.cat([node_features, neighbor_message], dim=1)
        transformed_features = self.transform(concatenated_features)

        if mask is None:
            mask = torch.ones(transformed_features.size(0), 1, device=node_features.device)
        mask[0, 0] = 0

        return transformed_features * mask, edge_hidden

class FullGraphEncoder(nn.Module):
    """Encodes a full graph into vector representations using embedded node and edge features."""
    def __init__(self, vocab, embedding_dim, num_layers):
        super(FullGraphEncoder, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.atom_feature_dim = vocab.vocabulary_size()
        self.bond_feature_dim = len(MoleculeGraph.BOND_TYPES)

        self.atom_embedding = torch.eye(vocab.vocabulary_size()).cuda()
        self.bond_embedding = torch.eye(self.bond_feature_dim).cuda()

        self.message_passing_encoder = MessagePassingEncoder(
            self.atom_feature_dim + self.bond_feature_dim, self.atom_feature_dim, embedding_dim, num_layers)

    def embed_features(self, graph_structures):
        node_ids, edge_ids, adj_matrix, bond_matrix, _ = graph_structures
        atom_features = self.atom_embedding[node_ids[:, 0]]

        edge_start_features = atom_features[edge_ids[:, 0]]
        edge_type_features = self.bond_embedding[edge_ids[:, 2]]
        concatenated_edge_features = torch.cat([edge_start_features, edge_type_features], dim=-1)

        return atom_features, concatenated_edge_features, adj_matrix, bond_matrix

    def forward(self, graph_structures):
        embedded_tensors = self.embed_features(graph_structures)
        node_outputs, _ = self.message_passing_encoder(*embedded_tensors)
        return node_outputs

    def encode_graph(self, graph_structures):
        """Encodes the graph structures and aggregates results."""
        node_vectors = self(graph_structures)
        # Combines node vectors based on graph partitioning information
        graph_vectors = combine_tensors([node_vectors[start: start + length] for start, length in graph_structures[-1]])
        return graph_vectors



