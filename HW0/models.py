from more_itertools import pairwise

"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
    
class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
        
class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=(96, 64),
                 sparse=False, embedding_sharing=True):
        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        self.U1 = ScaledEmbedding(num_users, embedding_dim=embedding_dim)
        self.Q1 = ScaledEmbedding(num_items, embedding_dim=embedding_dim)
        self.A1 = ZeroEmbedding(num_users, embedding_dim=1)
        self.B1 = ZeroEmbedding(num_items, embedding_dim=1)

        self.embedding_sharing = embedding_sharing
        if embedding_sharing:
            self.U2 = self.U1
            self.Q2 = self.Q1
        else:
            self.U2 = ScaledEmbedding(num_users, embedding_dim=embedding_dim)
            self.Q2 = ScaledEmbedding(num_items, embedding_dim=embedding_dim)

        layers = []
        for i, j in pairwise(layer_sizes):
            layers.append(nn.Linear(i, j))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], 1))
        self.mlp = nn.Sequential(*layers)

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        
    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of 
            shape (batch,). This corresponds to p_ij in the 
            assignment.
        score: tensor
            Tensor of user-item score predictions of shape 
            (batch,). This corresponds to r_ij in the 
            assignment.
        """
        
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        # getting embeddings:
        user_embd1 = self.U1(user_ids)
        item_embd1 = self.Q1(item_ids)
        user_bias1 = self.A1(user_ids).view(-1)
        item_bias1 = self.B1(item_ids).view(-1)

        # computing predictions:
        inner_products = (user_embd1 * item_embd1).sum(dim=1)
        predictions = inner_products + user_bias1 + item_bias1

        if self.embedding_sharing:
            user_embd2 = user_embd1
            item_embd2 = item_embd1
        else:
            user_embd2 = self.U2(user_ids)
            item_embd2 = self.Q2(item_ids)

        # computing scores:
        mlp_input = torch.cat([user_embd2, item_embd2, user_embd2 * item_embd2], dim=-1)
        score = self.mlp(mlp_input).view(-1)

        #********************************************************
        #********************************************************
        #********************************************************
        return predictions, score
