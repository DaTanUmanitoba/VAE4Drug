import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNDecoder(nn.Module):
    """
    A decoder model that uses GRU to generate sequences from latent vectors.
    """
    def __init__(self, vocab_dict, hidden_dim, latent_dim, layer_depth):
        super(RNNDecoder, self).__init__()
        self.vocab = vocab_dict
        self.vocab_map = {value: key for key, value in vocab_dict.items()}
        self.output_dim = len(vocab_dict)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.layer_depth = layer_depth

        self.rnn = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, num_layers=layer_depth, batch_first=False)
        self.fc_embed = nn.Linear(self.output_dim, latent_dim, bias=False)
        self.fc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, self.output_dim))
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def initialize_hidden(self, batch_size):
        initial_weights = next(self.parameters())
        return initial_weights.new_zeros(self.layer_depth, batch_size, self.hidden_dim)

    def forward(self, latent_vectors, sequence_targets, mask_targets):
        batch_size, sequence_length = sequence_targets.size()
        hidden_state = self.initialize_hidden(batch_size)
        latent_vectors = latent_vectors.unsqueeze(0)

        # Prepare inputs for RNN
        one_hot_targets = F.one_hot(sequence_targets, num_classes=self.output_dim).float().transpose(0, 1)
        one_hot_targets = torch.cat([torch.zeros([1, batch_size, self.output_dim], device=latent_vectors.device), one_hot_targets[:-1, :, :]])
        rnn_inputs = latent_vectors.repeat(sequence_length, 1, 1) + self.fc_embed(one_hot_targets)

        # RNN output
        rnn_output, hidden_state = self.rnn(rnn_inputs, hidden_state)
        output_predictions = self.fc_output(rnn_output).transpose(0, 1)

        # Loss computation
        losses = self.loss_fn(input=output_predictions.transpose(1, 2), target=sequence_targets)
        masked_losses = losses * mask_targets
        average_loss = torch.sum(masked_losses) / torch.sum(mask_targets)

        # Decoding predictions to sequences
        decoded_sequences = self.decode_outputs(output_predictions)
        return average_loss, decoded_sequences

    def decode_single(self, output_indices):
        """ Decode a single sequence """
        sequence_str = ''
        stop_token = self.vocab['[nop]']
        for idx in output_indices:
            if idx == stop_token:
                break
            sequence_str += self.vocab_map[idx]
        return sequence_str

    def decode(self, latent_vectors, max_length=20):
        """
        Decodes the latent vectors into sequences without reference sequences.
        """
        batch_size = latent_vectors.size(0)
        hidden_state = self.initialize_hidden(batch_size)
        latent_vectors = latent_vectors.unsqueeze(0)
        current_prediction = torch.zeros([1, batch_size, self.output_dim], device=latent_vectors.device)

        outputs = []
        for _ in range(max_length):
            rnn_input = latent_vectors + self.fc_embed(current_prediction)
            rnn_output, hidden_state = self.rnn(rnn_input, hidden_state)
            output_prediction = self.fc_output(rnn_output)
            probabilities = torch.softmax(output_prediction, dim=2)
            max_indices = torch.argmax(probabilities, dim=2)
            current_prediction = F.one_hot(max_indices, num_classes=self.output_dim).float()
            outputs.append(output_prediction)
        
        outputs = torch.cat(outputs, dim=0).transpose(0, 1)
        decoded_sequences = self.decode_outputs(outputs)
        return decoded_sequences

    def decode_outputs(self, predictions):
        """
        Helper function to decode batch of predictions into sequences.
        """
        output_indices = torch.argmax(predictions, dim=2).detach().cpu().numpy()
        sequences = [self.decode_single(indices) for indices in output_indices]
        return sequences
