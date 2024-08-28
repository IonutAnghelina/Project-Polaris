import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention

class EncoderBlock(nn.Module):

    def __init__(self, embedding_dim : int = 512, expansion_factor : int = 4, no_heads : int = 8, p_dropout : float = 0.1):
        
        """
        This layer passes through an encoder block architecture a set of (query, key, value) vectors

        The construction receives as input:

        embedding_dim - int, the dimension of the embeddings in query, key, value
        
        expansion_factor - int, The ratio between the size of the hidden layer and the size of the input in the FeedForward submodule
        
        no_heads - int, the number of heads in Multi Head Attention
        
        p_dropout - float, the dropout probability in the fully connected layers

        """
        
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embedding_dim, no_heads)

        self.firstLayerNorm = nn.LayerNorm(embedding_dim)
        self.secondLayerNorm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * expansion_factor),
            nn.ReLU(),
            nn.Linear(embedding_dim * expansion_factor, embedding_dim)
        )

        self.dropout1 = nn.Dropout(p_dropout)
        self.dropout2 = nn.Dropout(p_dropout)

    def forward(self, key : torch.Tensor, query : torch.Tensor, value : torch.Tensor,  mask : torch.Tensor = None, padding_mask : torch.Tensor = None) -> torch.Tensor:
        
        """
        Receives as input:

        key, value - [batch_size, seq_length, embedding_dim]
        
        query - [batch_size, query_seq_length, embedding_dim]

        The tensors that will be projected into the three spaces

        mask - [batch_size, key_seq_length, key_seq_length]

        The boolean mask that will be used in the case of masked self-attention

        padding_mask - [batch_size, query_seq_length, seq_length]

        The padding mask which will be used in order for padding tokens to not attend to other actual tokens

        Returns

        val - [batch_size, query_seq_len, embedding_dim] 

        The result obtained by passing the self-attention results through the Fully Connected and LayerNorm layers

        """
        
        attentionOutput = self.attention(key, query, value, mask, padding_mask)
        attentionDropoutResNorm = self.firstLayerNorm(value + self.dropout1(attentionOutput))
        
        fcOutput = self.fc(attentionDropoutResNorm)

        fcOutputDropoutResNorm = self.secondLayerNorm(attentionDropoutResNorm + self.dropout2(fcOutput))

        return fcOutputDropoutResNorm
        