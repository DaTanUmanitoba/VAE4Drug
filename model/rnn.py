import torch
import torch.nn as nn
from model.utils import select_indices

class CustomLSTM(nn.Module):
    """
    Custom implementation of an LSTM for the aggregate function of a MPGNN.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(CustomLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers

        # Gates with linear transformation followed by sigmoid activation (input, output, forget) and tanh (update)
        self.input_gate = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.Sigmoid())
        self.output_gate = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.Sigmoid())
        self.forget_gate = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.Sigmoid())
        self.update_gate = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.Tanh())

    def initial_state(self, features, state=None):
        """
        Generates initial hidden and cell states for an LSTM.
        """
        h_state = torch.zeros(len(features), self.hidden_dim, device=features.device)
        c_state = torch.zeros(len(features), self.hidden_dim, device=features.device)
        if state != None:
            h_state = torch.cat((h_state, state), dim=0)
            c_state = torch.cat((c_state, torch.zeros_like(state)), dim=0)
        return h_state, c_state

    def state_projection(self, h):
        return h[0]


    
    def compute_lstm(self, input_vector, hidden_states, cell_states):
        """
        Executes an LSTM cell update given the current input, adjacent hidden states, and corresponding cell states.
        """
        # Aggregate hidden states from neighbors
        combined_hidden = hidden_states.sum(dim=1)
        # Replicate input across the same dimensions as hidden states for concatenation
        expanded_input = input_vector.unsqueeze(1).expand(-1, hidden_states.size(1), -1) 
        
        # Compute gates for LSTM update
        input_gate = self.input_gate(torch.cat([input_vector, combined_hidden], dim=-1))
        output_gate = self.output_gate(torch.cat([input_vector, combined_hidden], dim=-1))
        forget_gate = self.forget_gate(torch.cat([expanded_input, hidden_states], dim=-1))
        update_gate = self.update_gate(torch.cat([input_vector, combined_hidden], dim=-1))
        
        # Update cell state
        new_cell_state = input_gate * update_gate + (forget_gate * cell_states).sum(dim=1)
        # Compute new hidden state
        new_hidden_state = output_gate * torch.tanh(new_cell_state)
        
        return new_hidden_state, new_cell_state


    def forward(self, features, adjacency):
        """
        Propagates through the LSTM layers over the depth of the network using the input features and adjacency matrix.
        """
        h_state = torch.zeros(features.size(0), self.hidden_dim, device=features.device)
        c_state = torch.zeros(features.size(0), self.hidden_dim, device=features.device)
        message_mask = torch.ones(h_state.size(0), 1, device=h_state.device)
        message_mask[0, 0] = 0 

        for layer in range(self.num_layers):
            h_neighbors = select_indices(h_state, 0, adjacency)
            c_neighbors = select_indices(c_state, 0, adjacency)
            h_state, c_state = self.compute_lstm(features, h_neighbors, c_neighbors)
            h_state *= message_mask
            c_state *= message_mask
        return h_state, c_state
