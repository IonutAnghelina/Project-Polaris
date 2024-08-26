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

        for pos in range(self.seq_size):
            for i in range(0,self.embedding_dim,2):
                self.positionalEmbeddings[pos,i] = np.sin(pos/((10000)**(2*i/self.embedding_dim)))
            for i in range(1,self.embedding_dim,2):
                self.positionalEmbeddings[pos,i] = np.cos(pos/((10000)**(2*i/self.embedding_dim)))

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        """
        Receives:

        x - torch.Tensor([seq_size, embedding_dim])

        The positionalEmbeddings tensor is added to x in order to enrich it with positional information
        """
        pos_emb = self.positionalEmbeddings.unsqueeze(0) 

        return x + pos_emb[:, :x.size(1), :]

