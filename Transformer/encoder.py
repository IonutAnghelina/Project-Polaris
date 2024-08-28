import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from embedding_layer import EmbeddingLayer
from positional_encoder import PositionalEncoder
from encoder_block import EncoderBlock

class Encoder(nn.Module):
    def __init__(self, seq_len : int, vocab_size : int, embedding_dim : int = 512, no_blocks : int = 6, expansion_factor : int = 4, no_heads : int = 8, p_dropout : float = 0.1):
        super(Encoder,self).__init__()
        
        """
        This block successively passes the (query, key, value) vectors through multiple encoder blocks

        The constructor receives as input:

        seq_len - int, the length of the input sequence

        vocab_size - int, the number of different tokens in the vocabulary
        
        embeddimg_dim - int, the embedding size of the vectors

        no_blocks - int, the number of the encoder blocks that will be applied to the input 

        expansion_factor - int, The ratio between the size of the hidden layer and the size of the input in the FeedForward submodule
        
        no_heads - int, the number of heads in the Multi Headed Attention
        
        p_dropout - float, the dropout probability in the fully connected layers
        """
        self.embedding_dim = embedding_dim
        self.no_heads = no_heads
        self.p_dropout = p_dropout
        self.expansion_factor = expansion_factor
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding_layer = EmbeddingLayer(self.vocab_size, self.embedding_dim)
        self.positional_embedder = PositionalEncoder(self.seq_len, self.embedding_dim)
        self.no_blocks = no_blocks
        self.dropout_layer = nn.Dropout(self.p_dropout)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(self.embedding_dim, self.expansion_factor, self.no_heads, self.p_dropout) for i in range(self.no_blocks)])
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        """
        Receives as input

        x - [batch_size, seq_len], the input sequence as token indices 

        Returns

        val - [batch_size, seq_len, embedding_dim] - the new embeddings for the tokens in the input

        """
        
        mask = None 
        
        padding_mask = (x!=0).int() # We identify the padding tokens and compute the padding mask
        
        batch_size = x.shape[0]
        
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.no_heads, self.seq_len, self.seq_len)
  
        
        inputTensor = self.embedding_layer(x)
        inputTensor = self.positional_embedder(inputTensor)
        inputTensor = self.dropout_layer(inputTensor) #We apply the embedding and positional encoder layers
        
        for block in self.encoder_blocks: #We successively apply all the blocks to the input tensor
            inputTensor = block(inputTensor, inputTensor, inputTensor, mask, padding_mask)

        return inputTensor
        