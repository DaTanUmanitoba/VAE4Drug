import torch
from torch.autograd import Function
from torch import nn
import numpy as np

class EmbeddingMatcherFunction(Function):
    """
    A custom autograd Function to find the nearest embeddings for a set of input vectors.
    This function computes the nearest embedding for each vector in a batch from a provided embedding matrix (the codebook).
    It also handles the backward pass manually to compute gradients with respect to the input vectors and embeddings.
    """

    @staticmethod
    def forward(ctx, inputs, embeddings):
        """
        Args:
            ctx: Context object that can be used to stash information for backward computation.
            inputs: Tensor of input vectors
            embeddings: Tensor of all possible embeddings: latent_dim x num_embeddings
        Returns:
            Tensor of closest embeddings corresponding to each input vector and indices of these embeddings in the embedding matrix.
        """
        if inputs.size(1) != embeddings.size(0):
            raise RuntimeError(f'input feature size {inputs.size(1)} must match embeddings size {embeddings.size(0)}')

        ctx.dims = list(range(len(inputs.size())))
        # Broadcasting inputs against embeddings to compute distances
        expanded_inputs = inputs.unsqueeze(-1)
        expanded_embeddings = embeddings.unsqueeze(1)
        #find the closest neighbour embedding
        distances = torch.norm(expanded_inputs - expanded_embeddings, 2, 1)
        closest_indices = distances.min(-1)[1]
        new_shape = [inputs.shape[0], *list(inputs.shape[2:]), inputs.shape[1]]
        #print(f'shalges: {embeddings.t().shape}  {closest_indices.view(-1).shape}   {new_shape}')
        closest_embeddings = embeddings.t().index_select(0, closest_indices.view(-1) ).view(new_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(closest_indices, inputs, embeddings )
        return closest_embeddings.contiguous(), closest_indices
    
    @staticmethod
    def backward(ctx, grad_output, closest_indices=None):
        """       
        Args:
            ctx: Context object with saved tensors.
            grad_output: Gradient of the loss with respect to the output of the forward pass.        
        Returns:
            Tuple of gradients with respect to the inputs and embeddings.
        """
        closest_indices, inputs, embeddings  = ctx.saved_tensors
        grad_inputs = grad_embeddings = None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output

        if ctx.needs_input_grad[1]:
            grad_embeddings = torch.zeros_like(embeddings)
            
            latent_inds = torch.arange(embeddings.size(1)).type_as(closest_indices)
            idx_choices = (closest_indices.view(-1, 1) == latent_inds.view(1, -1)).type_as(grad_output.data)
            n_choice = idx_choices.sum(0)
            n_choice[n_choice == 0] = 1
            choices = idx_choices / n_choice
            latent_num = int(np.prod(np.array(inputs.size()[2:])))
            embedding_dim, embedding_num = embeddings.shape
            
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(inputs.size(0) * latent_num, embedding_dim )
            grad_embeddings = torch.sum(grad_output.data.view(-1, embedding_dim, 1) * choices.view(-1, 1, embedding_num), 0)

        return grad_inputs, grad_embeddings


def closest_embedding_lookup(input_vectors, embedding_matrix):
    return EmbeddingMatcherFunction.apply(input_vectors, embedding_matrix)

class EmbeddingMatcher(nn.Module):
    def __init__(self, num_embeddings, embedding_size):
        super(EmbeddingMatcher, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(embedding_size, num_embeddings))

    def forward(self, feature_vectors, sg=False):
        """Retrieve the nearest embeddings for feature vectors from the embedding matrix codebook."""
        effective_embeddings = self.embeddings.detach() if sg else self.embeddings
        return closest_embedding_lookup(feature_vectors, effective_embeddings)
