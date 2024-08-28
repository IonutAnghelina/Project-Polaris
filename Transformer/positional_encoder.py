import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class PositionalEncoder(nn.Module):

    def __init__(self, seq_size : int, embedding_dim : int):

        """
        The class positional information to the tokens so that their order matters
        
        The constructor receives:

        seq_size - int, The length of the input sequence
        embedding_dim - int, The length of the embedding vectors
  
        The constructor builds:

        positionalEmbeddings - torch.Tensor([seq_size, embedding_dim])

        Each row pos in the tensor represents the positional Embedding of the Token at position pos.

        positionalEmbeddings[pos,i] = np.sin(pos/((10000)**(2*i/self.embedding_dim))) - for even i
        positionalEmbeddings[pos,i] = np.cos(pos/((10000)**(2*i/self.embedding_dim))) - for odd i 

        """
  
        super(PositionalEncoder, self).__init__()

        self.seq_size = seq_size 
        self.embedding_dim = embedding_dim

        self.positionalEmbeddings = torch.zeros(seq_size, embedding_dim)

        position = np.arange(seq_size).reshape(-1, 1)

       
        dimension = np.arange(embedding_dim)

        angle_rates = 1 / np.power(10000, (2 * (dimension // 2)) / embedding_dim)


        angles = position * angle_rates 

        self.positionalEmbeddings[:, 0::2] = torch.tensor(np.sin(angles[:, 0::2]), dtype=torch.float32)
        self.positionalEmbeddings[:, 1::2] = torch.tensor(np.cos(angles[:, 1::2]), dtype=torch.float32)

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        """
        Receives:

        x - torch.Tensor([seq_size, embedding_dim])

        The positionalEmbeddings tensor is added to x in order to enrich it with positional information
        """
        pos_emb = self.positionalEmbeddings.unsqueeze(0) 

        return x + pos_emb[:, :x.size(1), :]

