import torch
import torch.nn as nn
import torch.nn.functional as F
from selfies import decoder as selfies_decoder

from model.utils import combine_tensors
from model.mpgnn_encoder import FullGraphEncoder
from model.rnn_decoder import RNNDecoder
from model.auxiliary_predictor import MLP
from model.category_emb import EmbeddingMatcher

class VQVAE(nn.Module):
    def __init__(self, args):
        super(VQVAE, self).__init__()
        self.argsure(args)
        self.setup_modules(args)

    def argsure(self, args):
        """Initialize settings from args."""
        self.latent_dims = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.num_embed = args.num_embed
        
        self.atom_dict = args.atom_vocab
        self.selfies_dict = args.selfies_vocab
        
        self.vq_weight = args.vq_coef
        self.commit_weight = args.commit_coef
        self.prediction_weight = args.pred_coef
        self.recon_weight = args.recon_coef

    def setup_modules(self, args):
        self.initialize_network()
        self.encoder = FullGraphEncoder(args.atom_vocab, self.hidden_dim, args.agg_depth)
        self.decoder = RNNDecoder(args.selfies_vocab, self.hidden_dim, self.hidden_dim, args.rnn_depth)
        self.classifier = MLP(self.hidden_dim, 2)
        self.code_book = EmbeddingMatcher(self.num_embed, self.embedding_dim)

    def initialize_network(self):
        """Initialize network layers and weights."""
        self.embedding_dim = 2 * self.num_embed
        self.graph_transform = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
        self.encoder_output = nn.Linear(self.hidden_dim, self.embedding_dim * self.latent_dims)
        self.decoder_input = nn.Linear(self.embedding_dim * self.latent_dims, self.hidden_dim)

    def encode_input(self, input_data):
        """Encode input using the graph encoder."""
        encoded_output = self.encoder(input_data)
        encoded_output = [encoded_output[start:start+length] for start, length in input_data[-1]]
        return combine_tensors(encoded_output)

    def forward(self, input_data, selfies_data, targets, mask, labels):
        """Forward pass of the VQVAE."""
        input_data = [input.cuda().long() for input in input_data[:-1]] + [input_data[-1]]
        targets = torch.tensor(targets).cuda().long()
        mask = torch.tensor(mask).cuda().long()
        labels = torch.tensor(labels).cuda().long()
        
        encoded = self.encode_input(input_data)
        aggregated_encoded = torch.sum(encoded, dim=1)
        
        encoded = self.encoder_output(aggregated_encoded).reshape(-1, self.latent_dims, self.embedding_dim)
        quantized_embedding, _ = self.code_book(encoded.permute(0, 2, 1), sg=True)
        detached_embeddings, _ = self.code_book(encoded.permute(0, 2, 1).detach())
        quantized_output = self.decoder_input(quantized_embedding.permute(0, 2, 1).contiguous().view(-1, self.latent_dims * self.embedding_dim))
        
        reconstruction_loss, selfies_output = self.decoder(quantized_output, targets, mask)
        vq_loss = F.mse_loss(detached_embeddings, encoded.permute(0, 2, 1).detach())
        commitment_loss = F.mse_loss(encoded.permute(0, 2, 1), detached_embeddings.detach())
        prediction_loss, logits = self.classifier(quantized_output, labels)

        total_loss = (self.recon_weight * reconstruction_loss + self.vq_weight * vq_loss +
                      self.commit_weight * commitment_loss ) # + self.prediction_weight * prediction_loss)

        return total_loss, self.collect_stats(total_loss, reconstruction_loss, vq_loss, commitment_loss, prediction_loss), selfies_output, logits, selfies_data, labels

    def collect_stats(self, total_loss, reconstruction_loss, vq_loss, commitment_loss, prediction_loss):
        """Collect and format loss stats."""
        return {
            'total_loss': total_loss.item(),
            'recon_loss': reconstruction_loss.item(),
            'vq_loss': vq_loss.item(),
            'commit_loss': commitment_loss.item(),
            'prediction_loss': prediction_loss.item()
        }

    def decode_action(self, input, threshold=0.5):
        """
        Decode random sample from the latent space and filter outputs based on classifier predictions.
        Args: categorical latents tensor representing the embedded indices.
        Returns: list of decoded and filtered SMILES strings.
        """
        batch_size = input.shape[0]
        with torch.no_grad():
            # Reshape input and pass through decoder
            embedded = self.code_book.embeddings.t().index_select(0, input.view(-1)).view([batch_size, self.latent_dims * self.embedding_dim])
            latent_vectors = self.decoder_input(embedded.contiguous().view(-1, self.latent_dims * self.embedding_dim))
            
            # Decode selfies to SMILES
            decoded_selfies = self.decoder.decode(latent_vectors, max_length=20)
            
            # Predict and apply threshold filtering
            dummy_labels = torch.ones(batch_size, dtype=torch.long, device=input.device)
            _, logits = self.classifier(latent_vectors, dummy_labels)
            prediction_probs = torch.sigmoid(logits)[:, 1]  # Assuming binary classification
            
            # Collect only high confidence samples
            smiles_list = []
            for index, (selfie, prob) in enumerate(zip(decoded_selfies, prediction_probs)):
                if prob >= threshold:
                    decoded_smiles = selfies_decoder(selfie)
                    if decoded_smiles:
                        smiles_list.append('*' + decoded_smiles)

            return smiles_list
