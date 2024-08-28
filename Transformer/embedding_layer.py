import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size : int, embed_dim : int):

        """
        This layer converts each token, represented by an integer
        To a deep embedding vector

        The constructor receives as input:

        vocab_size - int, the size of the token vocabulary
        embed_dim - int, the size of the embedding vectors for each token 

        """
        super(EmbeddingLayer, self).__init__()

        self.model = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Receives:

        x - [batch_size, seq_length], the tensor of token indices

        seq_length is the length of the token sequence

        Returns a tensor of shape [batch_size, seq_length, embedding_dim]
        """
        return self.model(x)