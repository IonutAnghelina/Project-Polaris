import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from embedding_layer import EmbeddingLayer
from positional_encoder import PositionalEncoder
from decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, seq_len : int, target_vocab_size : int, embedding_dim : int = 512, no_blocks : int = 6, expansion_factor : int = 4, no_heads : int = 8, p_dropout : int = 0.1):
        super(Decoder,self).__init__()

        """
        This block successively passes the (query, key, value) vectors through multiple encoder blocks

        The constructor receives as input:

        seq_len - int, the length of the input sequence to the decoder
        
        target_vocab_size - int, the number of different tokens in the output vocabulary

        embeddimg_dim - int, the embedding size of the vectors

        no_blocks - int, the number of the decoder blocks that will be applied to the input

        expansion_factor - int, The ratio between the size of the hidden layer and the size of the input in the FeedForward submodule

        no_heads - int, the number of heads in the Multi Headed Attention
        
        p_dropout - float, the dropout probability in the fully connected layers

        """
        self.embedding_dim = embedding_dim
        self.no_heads = no_heads
        self.p_dropout = p_dropout
        self.expansion_factor = expansion_factor
        self.target_vocab_size = target_vocab_size
        self.seq_len = seq_len
        self.embedding_layer = EmbeddingLayer(self.target_vocab_size, self.embedding_dim)
        self.positional_embedder = PositionalEncoder(self.seq_len, self.embedding_dim)
        self.no_blocks = no_blocks
        self.fc = nn.Linear(self.embedding_dim, self.target_vocab_size)
        self.dropout_layer = nn.Dropout(self.p_dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(self.embedding_dim, self.no_heads, self.expansion_factor, self.p_dropout) for i in range(self.no_blocks)])
   
    def forward(self, x : torch.Tensor, encoder_output : torch.Tensor, mask : torch.Tensor, cross_attention_mask : torch.Tensor) -> torch.Tensor:
        
        """
        Receives as input:

        x - [batch_size, seq_len, embedding_size], the input tensor to the decoder 

        encoder_output - [batch_size, input_seq_len, embedding_size] - the output of the encoder block

        mask - [batch_size, seq_len, seq_len] - the attention mask that block future tokens attending to current tokens

        cross_attention_mask - [batch_size, seq_len, input_seq_len] - block padding tokens in the input sequence to attend

        Returns:

        out - [batch_size, seq_len, vocab_size] - the probabilities for each token to follow each prefix

        """
        padding_mask = (x!=0).int()
        batch_size = x.shape[0]
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.no_heads, self.seq_len, self.seq_len)
        
        inputTensor = self.embedding_layer(x)
        inputTensor = self.positional_embedder(inputTensor)
        inputTensor = self.dropout_layer(inputTensor)

        for block in self.decoder_blocks:
            inputTensor = block(inputTensor, inputTensor, inputTensor, encoder_output, encoder_output, encoder_output, mask, padding_mask, cross_attention_mask)
     
        inputTensor = self.fc(inputTensor)
      
        # return F.softmax(inputTensor, dim = -1)
        return inputTensor